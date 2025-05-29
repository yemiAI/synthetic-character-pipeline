from inference.generate import CharacterGenerator
from inference.consistency_check import ConsistencyChecker
import os

def test_full_generation():
    os.makedirs("test_outputs", exist_ok=True)
    gen = CharacterGenerator()
    images = []
    for i in range(3):
        img = gen.generate(f"in a test scene {i}", seed=123+i)
        path = f"test_outputs/out_{i}.png"
        img.save(path)
        images.append(path)

    checker = ConsistencyChecker()
    sim = checker.check(images)
    assert sim.mean().item() > 0.8
