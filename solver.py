import pickle

model=pickle.load(open('training_models/word2vec.pkl',"rb"))

a1 = "france"
a2 = "country"

b1 = "paris"

#print(model.wv.word_vec(a1))
#print(model.wv.word_vec(a2))

diff_a1 = model.wv.word_vec(a2) - model.wv.word_vec(a1)

sol_b1 = model.wv.word_vec(b1) + diff_a1




similar_list = (model.wv.similar_by_vector(sol_b1, topn = 6))
final_list = []

for word in similar_list:

	if word[0] == a1 or word[0] == a2 or word[0] == b1:
		continue
	final_list.append(word)

for item in final_list:
	print(item)
'''
if final_list[0][1] < 0.6:
	print("not able to find a solution")
else:
	print("word that solves the analogy is: " + final_list[0][0])

'''


