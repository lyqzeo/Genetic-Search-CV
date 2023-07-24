3

class MyClass:
    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        # Define custom behavior for "<"
        return self.value < other.value

    def __le__(self, other):
        # Define custom behavior for "<="
        return self.value <= other.value

    def __ge__(self, other):
        # Define custom behavior for ">="
        return self.value >= other.value

    def __eq__(self, other):
        # Define custom behavior for "=="
        return self.value == other.value

    def __ne__(self, other):
        # Define custom behavior for "!="
        return self.value != other.value