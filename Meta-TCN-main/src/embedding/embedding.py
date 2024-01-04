import datetime
from embedding.wordebd import WORDEBD

from model.triplet_contrast_model import ModelG


# 返回网络对象
def get_embedding(vocab, args):
    print("{}, Building embedding".format(
        datetime.datetime.now()), flush=True)

    # 创建Embedding对象
    ebd = WORDEBD(vocab, args.finetune_ebd)

    # 创建主网络实例
    modelG = ModelG(ebd, args)
    # modelD = ModelD(ebd, args)

    print("{}, Building embedding".format(
        datetime.datetime.now()), flush=True)

    if args.cuda != -1:
        modelG = modelG.cuda(args.cuda)
        # modelD = modelD.cuda(args.cuda)
        return modelG  # , modelD
    else:
        return modelG  # , modelD
    
