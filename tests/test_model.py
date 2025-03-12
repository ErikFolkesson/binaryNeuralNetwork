from src.models.model import MyModel
import torch

def test_model_initialization():
    model = MyModel()
    sample_input = torch.randn(1, 10)
    output = model(sample_input)
    assert output.shape == (1, 2)