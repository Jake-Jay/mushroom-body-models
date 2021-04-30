using Random, LinearAlgebra, Statistics, Logging
using Distributions, Parameters, JLD, BSON, Plots
using Flux, Zygote, Tracker, MLSuiteBase
using TrackerFlux, DistributedFlux, FastRecurFlux
using MAT

################################################################################
# utility functions

Flux.gpu(x) = x

Flux.adapt(T::Type{<:Real}, xs::UnitRange) = xs # prevent f32 conversion

∂(f, i = 1) = (xs...) -> Zygote.gradient(f, xs...)[i]

af(x, r0, θ) = max(0, r0)*max(tanh.(x - θ), zero(x)) # activation function

nargs(f) = methods(f).ms[1].nargs - 1

function *ᶜ(A, B)
    Cr = reshape(A, :, size(A)[end]) * reshape(B, size(B, 1), :)
    C = reshape(Cr, size(A)[1:end-1]..., size(B)[2:end]...)
end

macro staticvar(init)
    var = gensym()
    __module__.eval(:(const $var = $init))
    var = esc(var)
    quote
        global $var
        $var
    end
end

################################################################################
# f-I curve

fI(fs, xs...) = vcat([broadcast(f, [xs[n][i, :] for n in 1:nargs(f)]...) for (i, f) in fs]...)

∂fI(fs, xs...) = vcat([broadcast(∂(f), [xs[n][i, :] for n in 1:nargs(f)]...) for (i, f) in fs]...)

################################################################################
# CortexRNN

# follow notations in Song et al. (2016)
mutable struct CortexRNNCell{WI, WH, BH, WO, BO, H, M, Θ, F, T}
    Wi⁺::WI # input abs weight
    Wh⁺::WH # recur abs weight
    bh::BH  # recur bias
    Wo⁺::WO # output abs weight
    bo::BO  # output bias
    h::H    # initial hidden state
    Mi::M   # input mask (-1, 0, 1)
    Mh::M   # recur mask (-1, 0, 1)
    Mo::M   # output mask (-1, 0, 1)
    θs::Θ   # parameters of activation functions
    fs::F   # activation functions
    λΩ::T   # multiplier for vanishing-gradient regularization
    λH::T   # regularization of the h
    σi::T   # input noise
    σh::T   # recur noise
    α::T    # Δt / (τ = 100ms)
    dt::T   # integration step
end

function CortexRNNCell(;ni = 0, nh = 300, no = 0,
                Wi⁺ = nothing, Wh⁺ = nothing, bh = nothing, Wo⁺ = nothing,
                bo = nothing, h = nothing, Mi = ones(nh, ni), Mh = ones(nh, nh),
                Mo = ones(no, nh), fs = ((1:nh, af),), λΩ = 0f0, λH = 0f0, σi = 0f0,
                σh = 0f0, α = 0.2f0, dt = α, ρ = 1.5f0)
    @assert size(Mi) == (nh, ni)
    @assert size(Mh) == (nh, nh)
    @assert size(Mo) == (no, nh)
    Mi, Mh, Mo = f32.((Mi, Mh, Mo))
    #=
    if all(isone, Mh)
        rnd = rand(Bernoulli(0.8), nh)
        Mh = Mh * diagm(0 => 2rnd .- 1) # E:I ≈ 4:1
    end
    =#
    Wi⁺ = something(Wi⁺, param(rand(nh, ni)))
    Wh⁺ = something(Wh⁺, param(rand(nh, nh)))
    bh = something(bh, param(zeros(nh)))
    Wo⁺ = something(Wo⁺, param(rand(no, nh)))
    bo = something(bo, param(zeros(no)))
    #h = something(h, param(randn(nh, 1)))
    # rescale random matrix norm
    for z in [Wi⁺, Wh⁺, Wo⁺, h]
        rmul!(Tracker.data(z), ρ / opnorm(Tracker.data(z)))
    end
    θs = ntuple(i -> param((2-i)*ones(nh)), maximum(nargs, last.(fs)) - 1)
    CortexRNNCell(Wi⁺, Wh⁺, bh, Wo⁺, bo, h, Mi, Mh, Mo, θs, fs, λΩ, λH, σi, σh, α, dt) |> f32
end

Flux.hidden(m::CortexRNNCell) = m.h

Flux.@functor CortexRNNCell

(m::CortexRNNCell)(h) = h

function (m::CortexRNNCell)(h, x)
    @unpack Wi⁺, Wh⁺, bh, Wo⁺, bo, Mi, Mh, Mo = m
    @unpack fs, θs, λΩ, α, σi, σh, dt = m
    Wi = Mi .* Wi⁺
    Wh = Mh .* Wh⁺
    Wo = Mo .* Wo⁺
    dhdt = function (h)
        ξi = rmul!(randn!(similar(x)), √(2 / dt) * σi)
        ξh = rmul!(randn!(similar(h)), √(2 / dt) * σh)
        Δd = -h # + α * tanh.(h)
        Δi = Wi * (x .+ ξi)
        Δh = Wh * fI(fs, h, θs...) .+ bh
        return Δd .+ Δi .+ Δh .+ ξh
    end
    h′ = h
    for n in 1:(α / dt)
        h_ = h′ .+ dt .* dhdt(h′)
        h′ = Tracker.hook(h_) do Δ
            λΩ > 0 || return Δ
            ∂ε′ = Tracker.data(Δ)
            D = dt .* ∂fI(fs, Tracker.data(h′), Tracker.data.(θs)...)
            ∂ε = (1 - dt) * ∂ε′ .+ D .* (permutedims(Wh) *ᶜ ∂ε′)
            ∂ε_norm2 = sum(abs2.(∂ε), dims = 1)
            ∂ε′_norm2 = sum(abs2.(∂ε′), dims = 1)
            Ω = mean((∂ε_norm2 ./ ∂ε′_norm2 .- 1f0).^2)
            Tracker.back!(λΩ * Ω, once = false)
            return Δ
        end
    end
    o = Wo * fI(fs, h′, θs...) .+ bo
    return h′, (o, h′)
end

CortexRNN(a...; ka...) = Flux.Recur(CortexRNNCell(a...; ka...))

################################################################################
# loss function

function predict(model, x)
    model = model |> TrackerFlux.untrack
    xs = Flux.unstack(x, 2)
    ŷs = cpu.(first.(model.(gpu.(xs))))
    ŷ = Flux.stack(ŷs, 2)
    return ŷ
end

function loss(model, x, y, weight)
    @unpack λH = model.cell
    xs = gpu.(Flux.unstack(x, 2))
    ys = gpu.(Flux.unstack(y, 2))
    ws = weight[1, :, 1]
    os = model.(xs)
    ŷs = first.(os)
    hs = last.(os)
    Flux.reset!(model)
    l = mean(Flux.mse.(ys.*ws, ŷs.*ws))
    if λH > 0
        l += λH * sum(sum.(abs2, hs)) / sum(length.(hs))
    end
    return l
end

################################################################################
# gradient training

data_index = 6
println(string(data_index))
data = matread(string("../../data_3/", data_index, ".mat"))
x = data["x1"]
y = data["y"]
extend = 5
x = x[1:size(y, 1), :, :]
x = cat(repeat(reshape(x[:, 1, :], (size(x,1), 1, size(x,3))), 1, extend, 1), x, dims=2)
y = cat(repeat(reshape(y[:, 1, :], (size(y,1), 1, size(y,3))), 1, extend, 1), y, dims=2)
weight = cat(zeros(size(y,1), extend, size(y,3)),
    ones(size(y,1), size(y,2)-extend, size(y,3)), dims=2)
#load data

# create model
#root = get(ENV, "RNN_DATA", joinpath(@__DIR__, "..", "data"))

Mi = ones(size(x, 1), size(x, 1));

Mh = ones(size(x, 1), size(x, 1))
#Mh = Mh * diagm([ones(Int(round(size(x1, 1)*EI_ratio/(EI_ratio+1))));
#    -ones(Int(round(size(x1, 1)/(EI_ratio+1))))])
Mh = Mh - Diagonal(Mh)
Mo = Array([Array(Diagonal(ones(size(y, 1), size(y, 1))))';
    zeros(size(y, 1), size(x, 1)-size(y, 1))']');
h, bo = ones(size(x, 1), 1), zeros(size(y, 1))

ρ0 = 1.0

model = CortexRNN(
    ni = size(x, 1), nh = size(x, 1), no = size(y, 1), bo = bo, Mi = Mi,
    Mh = Mh, Mo = Mo, Wo⁺ = Mo, h = h, λΩ = 0f-3, α = 0.2f0, σh = 0.0f0, ρ = ρ0
) |> gpu

# train
ps = Flux.params(filter(Tracker.istracked, collect(Flux.params(model))))
opt = ADAMW(1f-3, (0.99, 0.999), 1f-4)
data = Flux.Data.DataLoader((x, y, weight), batchsize = 32, shuffle = true)
logger = DistributedFlux.TBLogger()
cb = Flux.throttle(1000) do
    #=
    ŷ = predict(model, x, y)
    plots = map(1:size(ŷ, 3)) do n
        plot(ŷ[:, :, n]')
    end
    =#
    params = Dict(name => tensor for (tensor, name) in namedparams(model))
    with_logger(logger) do
        @info "callback" params=params #plots=plots
    end
    BSON.@save "model.bson" model
end
@time Tracker.gradient(ps) do
    loss(model, first(data)...)
end
Flux.@epochs 3 Flux.train!(ps, data, opt, Tracker.gradient; cb, logger, verbose = true) do x, y, weight
    loss(model, x, y, weight)
end

model1 = Flux.params(model)
Wi_plus = model1[1]
Wh_plus = model1[2].data
bh = model1[3].data
Wo_plus = model1[4]
bo = model1[5]
h0 = model1[6]
Mi = model1[7]
Mh = model1[8]
Mo = model1[9]
r0 = model1[10].data
theta_s = model1[11].data

Base.Filesystem.mkpath("../result_local")
cd("../result_local")

file = matopen(string("model_", data_index, ".mat"), "w")

write(file, "Wi_plus", Wi_plus)
write(file, "Wh_plus", Wh_plus)
write(file, "bh", bh)
write(file, "Wo_plus", Wo_plus)
write(file, "bo", bo)
write(file, "h0", h0)
write(file, "Mi", Mi)
write(file, "Mh", Mh)
write(file, "Mo", Mo)
write(file, "r0", r0)
write(file, "theta_s", theta_s)

close(file)

pred_y = predict(model, x)

file = matopen(string("pred_y_", data_index, ".mat"), "w")

write(file, string("pred_y"), pred_y)

close(file)

cd("../src")
