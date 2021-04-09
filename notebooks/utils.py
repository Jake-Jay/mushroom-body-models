import re

def get_simple_name(name):
    return re.sub('[^A-Za-z]+', '', name.lower())

def greek_text_to_symbol(text):
    greek_text_to_symbol_dict={
        "alpha1": 'α1',
        "alpha2": 'α2',
        "alpha3": 'α3',
        "alpha'1": "α'1",
        "alpha'2": "α'2", 
        "alpha'3": "α'3",
        "beta1": 'β1',
        "beta2": 'β2',
        "beta'1": "β'1",
        "beta'2": "β'2",
        "gamma1": 'γ1',
        "gamma2": 'γ2',
        "gamma3": 'γ3',
        "gamma4": 'γ4',
        "gamma5": 'γ5'
    }
    return greek_text_to_symbol_dict[text]

def greek_symbol_to_text(symbol):
    greek_symbol_to_text_dict={
        'α1': "alpha1",
        'α2': "alpha2",
        'α3': "alpha3",
        "α'1": "alpha'1",
        "α'2": "alpha'2", 
        "α'3": "alpha'3",
        'β1': "beta1",
        'β2': "beta2",
        "β'1":"beta'1",
        "β'2": "beta'2",
        'γ1': "gamma1",
        'γ2': "gamma2",
        'γ3': "gamma3",
        'γ4': "gamma4",
        'γ5': "gamma5"
    }
    return greek_symbol_to_text_dict[symbol]


