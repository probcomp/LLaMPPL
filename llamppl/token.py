class Token:
    def __init__(self, token_id, token_str):
        self.token_id = token_id
        self.token_str = token_str
    
    # Support adding tokens to strings
    def __add__(self, other):
        if isinstance(other, Token):
            return self.token_str + other.token_str
        else:
            return self.token_str + other
    
    # Support adding strings to tokens
    def __radd__(self, other):
        return other + self.token_str
    
    # Support checking for EOS
    def __eq__(self, other):
        if isinstance(other, Token):
            return self.token_id == other.token_id
        elif isinstance(other, int):
            return self.token_id == other
        else:
            return self.token_str == other

    def __str__(self):
        return self.token_str