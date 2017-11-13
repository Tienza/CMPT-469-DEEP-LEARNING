import gensim

test_string = "Israeli Sniper Gets Only 45 Days for Killing of Two Women under White Flags: Israeli human rights group B'Tselem points out, the deal does not recognize guilt for the killing of Majedah and Rayah Abu Hajaj, but rather punishes the soldier for the \"killing of an unidentified individual.\""

model = gensim.models.Word2Vec(test_string, size=100, min_count=5)

print(model)