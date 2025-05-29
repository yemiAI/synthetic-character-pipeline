from inference.generate import CharacterGenerator

def test_single_image():
    gen = CharacterGenerator()
    img = gen.generate("in a library")
    assert img.size[0] > 0 and img.size[1] > 0
