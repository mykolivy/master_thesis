_REGISTRY = {}


def sequences():
	return _REGISTRY.copy()


def video_sequence(name=None):
	def decorate(cls):
		_REGISTRY[name] = cls
		return cls

	return decorate
