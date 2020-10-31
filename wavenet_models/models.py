
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'wavenet':
        from .wavenet_model import WavenetModel
        model = WavenetModel()
    else:
        raise ValueError("Model [%s] not recognized."% opt.model)

    model.initialize(opt)
    print("model [%s] is created" % (model.name()))

    return model
