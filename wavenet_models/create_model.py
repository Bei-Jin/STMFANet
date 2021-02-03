def create_model(opt):
    if opt.model == 'STMF':
        from .STMF_Model import STMFModel
        model = STMFModel()
    else:
        raise ValueError("Model [%s] not recognized."% opt.model)

    model.initialize(opt)
    print("model [%s] is created" % (model.name()))

    return model
