class NotEvaluatedError(Exception):
    def __str__(self):
        return "Node has not been evaluated yet! Call self.split() first."


class NotSupposedToHappenError(Exception):
    def __str__(self):
        return "This is just not supposed to happen."
