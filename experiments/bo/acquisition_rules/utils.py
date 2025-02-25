import torch
from sklearn.preprocessing import PowerTransformer

def power_transform_y(y):
    transformer = PowerTransformer(method='yeo-johnson')
    y = y.squeeze(0).double().clone()
    if y.std() > 1_000 or y.mean().abs() > 1_000:
        print('large y values, standardizing . . .')
        y = (y - y.mean()) / y.std()
    y_transformed = transformer.fit_transform(y)
    return torch.as_tensor(y_transformed).view_as(y)