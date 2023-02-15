import numpy as np
import pytest
import torch
from PIL import Image

import maskclip


@pytest.mark.parametrize('model_name', maskclip.available_models())
def test_consistency(model_name):
    device = "cpu"
    jit_model, transform = maskclip.load(model_name, device=device, jit=True)
    py_model, _ = maskclip.load(model_name, device=device, jit=False)

    image = transform(Image.open("MaskCLIP.png")).unsqueeze(0).to(device)
    text = maskclip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    with torch.no_grad():
        logits_per_class, _, _ = jit_model(image, text)
        jit_probs = logits_per_class.softmax(dim=-1).cpu().numpy()

        logits_per_class, _, _ = py_model(image, text)
        py_probs = logits_per_class.softmax(dim=-1).cpu().numpy()

    assert np.allclose(jit_probs, py_probs, atol=0.01, rtol=0.1)
