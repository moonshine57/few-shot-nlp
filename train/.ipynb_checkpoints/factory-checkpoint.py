import train.regular as regular

def train(train_data, val_data, model,args):
    return regular.train(train_data, val_data, model,args)

def test(test_data, model, args,episodes):
    return regular.test(test_data, model, args,episodes)
