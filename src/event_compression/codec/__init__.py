_REGISTER = {}

def codecs():
	return _REGISTER.copy()

def codec(name=None):
	def decorate(cls):
		_REGISTER[name] = cls
	return decorate