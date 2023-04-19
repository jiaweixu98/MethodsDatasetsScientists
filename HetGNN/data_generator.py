import numpy as np
import re
import random
from collections import Counter
from itertools import *

class input_data(object):
	def __init__(self, args):
		self.args = args
		# 不是每个p都有对应的a, p, b, d, m 是否可行？
		a_p_list_train = [[] for k in range(self.args.A_n)]
		p_a_list_train = [[] for k in range(self.args.P_n)]
		p_p_cite_list_train = [[] for k in range(self.args.P_n)]
		b_p_list_train = [[] for k in range(self.args.B_n)]
		p_b = [[] for k in range(self.args.P_n)]
		d_p_list_train = [[] for k in range(self.args.D_n)]
		p_d = [[] for k in range(self.args.P_n)]
		m_p_list_train = [[] for k in range(self.args.M_n)]
		p_m = [[] for k in range(self.args.P_n)]

		relation_f = ["a_p_list_train.txt", "p_a_list_train.txt", "p_p_citation_list.txt", "b_p_list_train.txt", 'p_b.txt', "d_p_list_train.txt", 'p_d.txt', "m_p_list_train.txt", 'p_m.txt', ]
		#store academic relational data 
		for i in range(len(relation_f)):
			f_name = relation_f[i]
			neigh_f = open(self.args.data_path + f_name, "r")
			for line in neigh_f:
				line = line.strip()
				node_id = int(re.split(':', line)[0])
				neigh_list = re.split(':', line)[1]
				neigh_list_id = re.split(',', neigh_list)
				if f_name == 'a_p_list_train.txt':
					for j in range(len(neigh_list_id)):
						a_p_list_train[node_id].append('p'+str(neigh_list_id[j]))
				elif f_name == 'p_a_list_train.txt':
					for j in range(len(neigh_list_id)):
						p_a_list_train[node_id].append('a'+str(neigh_list_id[j]))
				elif f_name == 'p_p_citation_list.txt':
					for j in range(len(neigh_list_id)):
						p_p_cite_list_train[node_id].append('p'+str(neigh_list_id[j]))
				elif f_name == 'p_b.txt':
					for j in range(len(neigh_list_id)):
						p_b[node_id].append('b'+str(neigh_list_id[j]))
				elif f_name == 'b_p_list_train.txt':
					for j in range(len(neigh_list_id)):
						b_p_list_train[node_id].append('p'+str(neigh_list_id[j]))
				elif f_name == 'p_d.txt':
					for j in range(len(neigh_list_id)):
						p_d[node_id].append('d'+str(neigh_list_id[j]))
				elif f_name == 'd_p_list_train.txt':
					for j in range(len(neigh_list_id)):
						d_p_list_train[node_id].append('p'+str(neigh_list_id[j]))
				elif f_name == 'p_m.txt':
					for j in range(len(neigh_list_id)):
						p_d[node_id].append('m'+str(neigh_list_id[j]))
				elif f_name == 'm_p_list_train.txt':
					for j in range(len(neigh_list_id)):
						d_p_list_train[node_id].append('p'+str(neigh_list_id[j]))
			neigh_f.close()


		#paper neighbor: author + citation + bio + dataset + method
		p_neigh_list_train = [[] for k in range(self.args.P_n)]
		for i in range(self.args.P_n):
			p_neigh_list_train[i] += p_a_list_train[i]
			p_neigh_list_train[i] += p_p_cite_list_train[i] 
			p_neigh_list_train[i] += p_b[i]
			p_neigh_list_train[i] += p_d[i]
			p_neigh_list_train[i] += p_m[i]


		self.a_p_list_train =  a_p_list_train
		self.p_a_list_train = p_a_list_train
		self.p_p_cite_list_train = p_p_cite_list_train
		self.p_neigh_list_train = p_neigh_list_train
		self.b_p_list_train = b_p_list_train
		self.d_p_list_train = d_p_list_train
		self.m_p_list_train = m_p_list_train
		# if ==2, do not load the pre trained data.
		if self.args.train_test_label != 2:
			self.triple_sample_p = self.compute_sample_p()
			#store paper content pre-trained embedding
			# 这里的ebd注意下,128维度，且第一行是行数+维度
			# 用abstract 代表所有的 embedding.
			p_abstract_embed = np.zeros((self.args.P_n, self.args.in_f_d))
			p_a_e_f = open(self.args.data_path + "p_abstract_embed.txt", "r")
			for line in islice(p_a_e_f, 1, None): # 这里跳过了第一行
				values = line.split()
				index = int(values[0])
				embeds = np.asarray(values[1:], dtype='float32')
				p_abstract_embed[index] = embeds
			p_a_e_f.close()

			self.p_abstract_embed = p_abstract_embed
			# we do not have this.
			self.p_title_embed = ''

			#store pre-trained network/content embedding
			a_net_embed = np.zeros((self.args.A_n, self.args.in_f_d))
			p_net_embed = np.zeros((self.args.P_n, self.args.in_f_d))
			# 注意维度
			b_net_embed = np.zeros((self.args.B_n, self.args.in_f_d))
			d_net_embed = np.zeros((self.args.D_n, self.args.in_f_d))
			m_net_embed = np.zeros((self.args.M_n, self.args.in_f_d))
			# 注意路径，这个net是Word2Vec得到的，待会儿要生成一下。deep walk的结果，每个节点都应该要有联系。
			net_e_f = open(self.args.data_path + "node_net_embedding.txt", "r")
			for line in islice(net_e_f, 1, None):
				line = line.strip()
				index = re.split(' ', line)[0]
				if len(index) and (index[0] == 'a' or index[0] == 'b' or index[0] == 'p' or index[0] == 'd' or index[0] == 'm'):
					embeds = np.asarray(re.split(' ', line)[1:], dtype='float32')
					if index[0] == 'a':
						a_net_embed[int(index[1:])] = embeds
					elif index[0] == 'b':
						b_net_embed[int(index[1:])] = embeds
					elif index[0] == 'p':
						p_net_embed[int(index[1:])] = embeds
					elif index[0] == 'd':
						d_net_embed[int(index[1:])] = embeds
					elif index[0] == 'm':
						m_net_embed[int(index[1:])] = embeds
			net_e_f.close()

			p_b_net_embed = np.zeros((self.args.P_n, self.args.in_f_d))
			for i in range(self.args.P_n):
				# 注意，此处有的 p_b_net_embed会为空！ 因为有些paper 没有 b。是否会导致bug? 
				if len(p_b[i]):
					for j in range(len(p_b[i])):
						b_id = int(p_b[i][j][1:])
						p_b_net_embed[i] = np.add(p_b_net_embed[i], b_net_embed[b_id])
					# 平均化处理，这里有一些是空的。因为没有。一个解决办法是学reference, 用本身的ebd代替。 发现自己的邻居和自己一摸一样？可能可以的。
					p_b_net_embed[i] = p_b_net_embed[i] / len(p_b[i])
				# so we add this. 
				else:
					p_b_net_embed[i] = p_net_embed[i]

			p_d_net_embed = np.zeros((self.args.P_n, self.args.in_f_d))
			for i in range(self.args.P_n):
				# 同上
				if len(p_d[i]):
					for j in range(len(p_d[i])):
						d_id = int(p_d[i][j][1:])
						p_d_net_embed[i] = np.add(p_d_net_embed[i], d_net_embed[d_id])
					p_d_net_embed[i] = p_d_net_embed[i] / len(p_d[i])
				else:
					p_d_net_embed[i] = p_net_embed[i]
			
			p_m_net_embed = np.zeros((self.args.P_n, self.args.in_f_d))
			for i in range(self.args.P_n):
				# 同上
				if len(p_m[i]):
					for j in range(len(p_m[i])):
						m_id = int(p_m[i][j][1:])
						p_m_net_embed[i] = np.add(p_m_net_embed[i], m_net_embed[m_id])
					p_m_net_embed[i] = p_m_net_embed[i] / len(p_m[i])
				else:
					p_m_net_embed[i] = p_net_embed[i]

			p_a_net_embed = np.zeros((self.args.P_n, self.args.in_f_d))
			for i in range(self.args.P_n):
				# 不同上，因为每篇文章都有author.
				if len(p_a_list_train[i]):
					for j in range(len(p_a_list_train[i])):
						a_id = int(p_a_list_train[i][j][1:])
						p_a_net_embed[i] = np.add(p_a_net_embed[i], a_net_embed[a_id])
					p_a_net_embed[i] = p_a_net_embed[i] / len(p_a_list_train[i])
			
			# 参考文献的embedding, 这个没有用上，原因是一部分没有参考文献？这样ref仅仅在随机游走时用到了。如果一篇文章没有参考文献，那么其参考文献embeddings就是他自己。（似乎也可以说的通。贡献了一部分新东西，因为有的文章有参考文献。）
			p_ref_net_embed = np.zeros((self.args.P_n, self.args.in_f_d))
			for i in range(self.args.P_n):
				if len(p_p_cite_list_train[i]):
					for j in range(len(p_p_cite_list_train[i])):
						p_id = int(p_p_cite_list_train[i][j][1:])
						p_ref_net_embed[i] = np.add(p_ref_net_embed[i], p_net_embed[p_id])
					p_ref_net_embed[i] = p_ref_net_embed[i] / len(p_p_cite_list_train[i])
				else:
					p_ref_net_embed[i] = p_net_embed[i]

			#empirically use 3 paper embedding for author content embeding generation
			a_text_embed = np.zeros((self.args.A_n, self.args.in_f_d * 3))
			for i in range(self.args.A_n):
				if len(a_p_list_train[i]):
					feature_temp = []
					if len(a_p_list_train[i]) >= 3:
						# id_list_temp = random.sample(a_p_list_train[i], 3)
						for j in range(3):
							feature_temp.append(p_abstract_embed[int(a_p_list_train[i][j][1:])])
					else:
						for j in range(len(a_p_list_train[i])):
							feature_temp.append(p_abstract_embed[int(a_p_list_train[i][j][1:])])
						for k in range(len(a_p_list_train[i]), 3):
							feature_temp.append(p_abstract_embed[int(a_p_list_train[i][-1][1:])])

					feature_temp = np.reshape(np.asarray(feature_temp), [1, -1])
					a_text_embed[i] = feature_temp

#empirically use 5 paper embedding for dataset content embeding generation
			d_text_embed = np.zeros((self.args.D_n, self.args.in_f_d * 5))
			for i in range(self.args.D_n):
				if len(d_p_list_train[i]):
					feature_temp = []
					if len(d_p_list_train[i]) >= 5:
						# id_list_temp = random.sample(a_p_list_train[i], 3)
						for j in range(5):
							feature_temp.append(p_abstract_embed[int(d_p_list_train[i][j][1:])])
					else:
						for j in range(len(d_p_list_train[i])):
							feature_temp.append(p_abstract_embed[int(d_p_list_train[i][j][1:])])
						for k in range(len(d_p_list_train[i]), 5):
							feature_temp.append(p_abstract_embed[int(d_p_list_train[i][-1][1:])])

					feature_temp = np.reshape(np.asarray(feature_temp), [1, -1])
					d_text_embed[i] = feature_temp
			# #empirically use 5 paper embedding for bioentity content embeding generation
			b_text_embed = np.zeros((self.args.B_n, self.args.in_f_d * 5))
			for i in range(self.args.B_n):
				if len(b_p_list_train[i]):
					feature_temp = []
					if len(b_p_list_train[i]) >= 5:
						# id_list_temp = random.sample(a_p_list_train[i], 5)
						for j in range(5):
							feature_temp.append(p_abstract_embed[int(b_p_list_train[i][j][1:])])
					else:
						for j in range(len(b_p_list_train[i])):
							feature_temp.append(p_abstract_embed[int(b_p_list_train[i][j][1:])])
						for k in range(len(b_p_list_train[i]), 5):
							feature_temp.append(p_abstract_embed[int(b_p_list_train[i][-1][1:])])

					feature_temp = np.reshape(np.asarray(feature_temp), [1, -1])
					b_text_embed[i] = feature_temp
					# #empirically use 5 paper embedding for method content embeding generation
			m_text_embed = np.zeros((self.args.M_n, self.args.in_f_d * 5))
			for i in range(self.args.M_n):
				# 同上
				if len(m_p_list_train[i]):
					feature_temp = []
					if len(m_p_list_train[i]) >= 5:
						# id_list_temp = random.sample(a_p_list_train[i], 5)
						for j in range(5):
							feature_temp.append(p_abstract_embed[int(m_p_list_train[i][j][1:])])
					else:
						for j in range(len(m_p_list_train[i])):
							feature_temp.append(p_abstract_embed[int(m_p_list_train[i][j][1:])])
						for k in range(len(m_p_list_train[i]), 5):
							feature_temp.append(p_abstract_embed[int(m_p_list_train[i][-1][1:])])

					feature_temp = np.reshape(np.asarray(feature_temp), [1, -1])
					m_text_embed[i] = feature_temp
			# this is node ebd, net ebd is from deepwalk. text embed from spector.
			# p_a net: consider p's direct author's ebd. etc.
			# It is important to consider a paper's author or its method, right?
			# If a paper does not have an author, just use the paper itself's embedding. Is that ok?
			self.p_b = p_b
			self.p_d = p_d
			self.p_b_net_embed = p_b_net_embed
			self.p_d_net_embed = p_d_net_embed
			self.p_a_net_embed = p_a_net_embed
			self.p_ref_net_embed = p_ref_net_embed
			self.p_net_embed = p_net_embed
			self.a_net_embed = a_net_embed
			self.a_text_embed = a_text_embed
			self.b_net_embed = b_net_embed
			self.b_text_embed = b_text_embed
			self.d_net_embed = d_net_embed
			self.d_text_embed = d_text_embed
			self.m_net_embed = m_net_embed
			self.m_text_embed = m_text_embed
			


			#store neighbor set from random walk sequence (with restart)
			a_neigh_list_train = [[[] for i in range(self.args.A_n)] for j in range(5)]
			p_neigh_list_train = [[[] for i in range(self.args.P_n)] for j in range(5)]
			b_neigh_list_train = [[[] for i in range(self.args.B_n)] for j in range(5)]
			d_neigh_list_train = [[[] for i in range(self.args.D_n)] for j in range(5)]
			m_neigh_list_train = [[[] for i in range(self.args.M_n)] for j in range(5)]
			# random walk (with restart results)
			het_neigh_train_f = open(self.args.data_path + "het_neigh_train.txt", "r")
			for line in het_neigh_train_f:
				line = line.strip()
				node_id = re.split(':', line)[0]
				neigh = re.split(':', line)[1]
				neigh_list = re.split(',', neigh)
				# determine what this line is. len(node_id) > 1 is useless here?
				if node_id[0] == 'a' and len(node_id) > 1:
					for j in range(len(neigh_list)):
						# the neighlist should not be empty (if not, may have bugs). because this is the result of Random walk with restart.
						if neigh_list[j][0] == 'a':
							a_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'p':
							a_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'b':
							a_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'd':
							a_neigh_list_train[3][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'm':
							a_neigh_list_train[4][int(node_id[1:])].append(int(neigh_list[j][1:]))
				elif node_id[0] == 'p' and len(node_id) > 1:
					for j in range(len(neigh_list)):
						if neigh_list[j][0] == 'a':
							p_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'p':
							p_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'b':
							p_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'd':
							p_neigh_list_train[3][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'm':
							p_neigh_list_train[4][int(node_id[1:])].append(int(neigh_list[j][1:]))
				elif node_id[0] == 'b' and len(node_id) > 1:
					for j in range(len(neigh_list)):
						if neigh_list[j][0] == 'a':
							b_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'p':
							b_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'b':
							b_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'd':
							b_neigh_list_train[3][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'm':
							b_neigh_list_train[4][int(node_id[1:])].append(int(neigh_list[j][1:]))
				elif node_id[0] == 'd' and len(node_id) > 1:
					for j in range(len(neigh_list)):
						if neigh_list[j][0] == 'a':
							d_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'p':
							d_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'b':
							d_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'd':
							d_neigh_list_train[3][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'm':
							d_neigh_list_train[4][int(node_id[1:])].append(int(neigh_list[j][1:]))
				elif node_id[0] == 'm' and len(node_id) > 1:
					for j in range(len(neigh_list)):
						if neigh_list[j][0] == 'a':
							m_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'p':
							m_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'b':
							m_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'd':
							m_neigh_list_train[3][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'm':
							m_neigh_list_train[4][int(node_id[1:])].append(int(neigh_list[j][1:]))
			het_neigh_train_f.close()
			#print a_neigh_list_train[0][1]

			#store top neighbor set (based on frequency) from random walk sequence
			a_neigh_list_train_top = [[[] for i in range(self.args.A_n)] for j in range(5)]
			p_neigh_list_train_top = [[[] for i in range(self.args.P_n)] for j in range(5)]
			b_neigh_list_train_top = [[[] for i in range(self.args.B_n)] for j in range(5)]
			d_neigh_list_train_top = [[[] for i in range(self.args.D_n)] for j in range(5)]
			m_neigh_list_train_top = [[[] for i in range(self.args.M_n)] for j in range(5)]
			top_k = [10, 10, 10, 3, 3] #fix each neighor type size, becasue the dataset and method is relatively rare.
			# they should not be empty!
			for i in range(self.args.A_n):
				for j in range(5):
					a_neigh_list_train_temp = Counter(a_neigh_list_train[j][i])
					top_list = a_neigh_list_train_temp.most_common(top_k[j])
					neigh_size = 0
					if j == 3 or j == 4:
						neigh_size = 3
					else:
						neigh_size = 10
					for k in range(len(top_list)):
						a_neigh_list_train_top[j][i].append(int(top_list[k][0]))
					# if it is not empty.
					if len(a_neigh_list_train_top[j][i]) and len(a_neigh_list_train_top[j][i]) < neigh_size:
						for l in range(len(a_neigh_list_train_top[j][i]), neigh_size):
							a_neigh_list_train_top[j][i].append(random.choice(a_neigh_list_train_top[j][i]))

			for i in range(self.args.P_n):
				for j in range(5):
					p_neigh_list_train_temp = Counter(p_neigh_list_train[j][i])
					top_list = p_neigh_list_train_temp.most_common(top_k[j])
					neigh_size = 0
					if j == 3 or j == 4:
						neigh_size = 3
					else:
						neigh_size = 10
					for k in range(len(top_list)):
						p_neigh_list_train_top[j][i].append(int(top_list[k][0]))
					if len(p_neigh_list_train_top[j][i]) and len(p_neigh_list_train_top[j][i]) < neigh_size:
						for l in range(len(p_neigh_list_train_top[j][i]), neigh_size):
							p_neigh_list_train_top[j][i].append(random.choice(p_neigh_list_train_top[j][i]))

			for i in range(self.args.B_n):
				for j in range(5):
					b_neigh_list_train_temp = Counter(b_neigh_list_train[j][i])
					top_list = b_neigh_list_train_temp.most_common(top_k[j])
					neigh_size = 0
					if j == 3 or j == 4 :
						neigh_size = 3
					else:
						neigh_size = 10
					for k in range(len(top_list)):
						b_neigh_list_train_top[j][i].append(int(top_list[k][0]))
					if len(b_neigh_list_train_top[j][i]) and len(b_neigh_list_train_top[j][i]) < neigh_size:
						for l in range(len(b_neigh_list_train_top[j][i]), neigh_size):
							b_neigh_list_train_top[j][i].append(random.choice(b_neigh_list_train_top[j][i]))
			
			for i in range(self.args.D_n):
				for j in range(5):
					d_neigh_list_train_temp = Counter(d_neigh_list_train[j][i])
					top_list = d_neigh_list_train_temp.most_common(top_k[j])
					neigh_size = 0
					if j == 3 or j == 4:
						neigh_size = 3
					else:
						neigh_size = 10
					for k in range(len(top_list)):
						d_neigh_list_train_top[j][i].append(int(top_list[k][0]))
					if len(d_neigh_list_train_top[j][i]) and len(d_neigh_dlist_train_top[j][i]) < neigh_size:
						for l in range(len(d_neigh_list_train_top[j][i]), neigh_size):
							d_neigh_list_train_top[j][i].append(
								random.choice(d_neigh_list_train_top[j][i]))
							
			for i in range(self.args.M_n):
				for j in range(5):
					m_neigh_list_train_temp = Counter(m_neigh_list_train[j][i])
					top_list = m_neigh_list_train_temp.most_common(top_k[j])
					neigh_size = 0
					if j == 3 or j == 4:
						neigh_size = 3
					else:
						neigh_size = 10
					for k in range(len(top_list)):
						m_neigh_list_train_top[j][i].append(int(top_list[k][0]))
					if len(m_neigh_list_train_top[j][i]) and len(m_neigh_list_train_top[j][i]) < neigh_size:
						for l in range(len(m_neigh_list_train_top[j][i]), neigh_size):
							m_neigh_list_train_top[j][i].append(
								random.choice(m_neigh_list_train_top[j][i]))

			a_neigh_list_train[:] = []
			p_neigh_list_train[:] = []
			b_neigh_list_train[:] = []
			d_neigh_list_train[:] = []
			m_neigh_list_train[:] = []

			self.a_neigh_list_train = a_neigh_list_train_top
			self.p_neigh_list_train = p_neigh_list_train_top
			self.b_neigh_list_train = b_neigh_list_train_top
			self.d_neigh_list_train = d_neigh_list_train_top
			self.m_neigh_list_train = m_neigh_list_train_top






















			#store ids of author/paper/bio/dataset/method used in training
			train_id_list = [[] for i in range(5)]
			for i in range(5):
				if i == 0:
					for l in range(self.args.A_n):
						if len(a_neigh_list_train_top[i][l]):
							train_id_list[i].append(l)
					self.a_train_id_list = np.array(train_id_list[i])
				elif i == 1:
					for l in range(self.args.P_n):
						if len(p_neigh_list_train_top[i][l]):
							train_id_list[i].append(l)
					self.p_train_id_list = np.array(train_id_list[i])
				elif i == 2:
					for l in range(self.args.B_n):
						if len(b_neigh_list_train_top[i][l]):
							train_id_list[i].append(l)
					self.b_train_id_list = np.array(train_id_list[i])
				elif i == 3:
					for l in range(self.args.D_n):
						if len(d_neigh_list_train_top[i][l]):
							train_id_list[i].append(l)
					self.d_train_id_list = np.array(train_id_list[i])
				elif i == 4:
					for l in range(self.args.M_n):
						if len(m_neigh_list_train_top[i][l]):
							train_id_list[i].append(l)
					self.m_train_id_list = np.array(train_id_list[i])

			#重启随机游走
	def het_walk_restart(self):
		a_neigh_list_train = [[] for k in range(self.args.A_n)]
		p_neigh_list_train = [[] for k in range(self.args.P_n)]
		b_neigh_list_train = [[] for k in range(self.args.B_n)]
		d_neigh_list_train = [[] for k in range(self.args.D_n)]
		m_neigh_list_train = [[] for k in range(self.args.M_n)]
		#generate neighbor set via random walk with restart
		node_n = [self.args.A_n, self.args.P_n, self.args.B_n, self.args.D_n, self.args.M_n]
		for i in range(5):
			for j in range(node_n[i]):
				if i == 0:
					# temp是从第一个节点开始抽
					neigh_temp = self.a_p_list_train[j]
					# train是选一个对应位置的空白列表，往里放东西 (指针)
					neigh_train = a_neigh_list_train[j]
					curNode = "a" + str(j)
				elif i == 1:
					# paper 从 author开始，默认 paper必须有作者，不然没意义。我们需要在这上边进行筛选。
					neigh_temp = self.p_a_list_train[j]
					neigh_train = p_neigh_list_train[j]
					curNode = "p" + str(j)
				elif i == 2:
					neigh_temp = self.b_p_list_train[j]
					neigh_train = b_neigh_list_train[j]
					curNode = "b" + str(j)
				elif i == 3:
					neigh_temp = self.d_p_list_train[j]
					neigh_train = d_neigh_list_train[j]
					curNode = "d" + str(j)
				elif i == 4:
					neigh_temp = self.m_p_list_train[j]
					neigh_train = m_neigh_list_train[j]
					curNode = "m" + str(j)
					# 此处表示该类型的节点是存在的，才会在后续考虑
				if len(neigh_temp):
					# flag_curNode = curNode
					# [删除了这些条件，因为我们有dataset，有底气] 加一些限制条件，假如返回最初的起点超过10次，说明大概率找不到100个节点了，就continue
					# back_count = 0
					neigh_L = 0
					a_L = 0
					p_L = 0
					b_L = 0
					d_L = 0
					m_L = 0
					while neigh_L < 110: #maximum neighbor size = 110,(apbdm) 30,30,30,10,10
						rand_p = random.random() #return p； 返回的概率调小了，多走两步吧。
						if rand_p > 0.4:
							if curNode[0] == "a":
								# find a neighbor p.
								curNode = random.choice(self.a_p_list_train[int(curNode[1:])])
								if p_L < 30: #size constraint (make sure each type of neighobr is sampled)
									neigh_train.append(curNode)
									neigh_L += 1
									p_L += 1
							elif curNode[0] == "p":
								# self.p_neigh_list_train != p_neigh_list_train; find any type, at least it is author node.
								curNode = random.choice(self.p_neigh_list_train[int(curNode[1:])])
								# avoid go back to the same author. so many authors.
								if curNode != ('a' + str(j)) and curNode[0] == 'a' and a_L < 30:
									neigh_train.append(curNode)
									neigh_L += 1
									a_L += 1
								# avoid go back to the same identities.
								elif curNode != ('b' + str(j)) and curNode[0] == 'b' and b_L < 30:
									neigh_train.append(curNode)
									neigh_L += 1
									b_L += 1
								elif curNode[0] == 'd':
									if d_L < 10:
										neigh_train.append(curNode)
										neigh_L += 1
										d_L += 1
								elif curNode[0] == 'm':
									if m_L < 10:
										neigh_train.append(curNode)
										neigh_L += 1
										m_L += 1
							elif curNode[0] == "b":
								curNode = random.choice(self.b_p_list_train[int(curNode[1:])])
								if p_L < 30:
									neigh_train.append(curNode)
									neigh_L +=1
									p_L += 1
							elif curNode[0] == "d":
								curNode = random.choice(self.d_p_list_train[int(curNode[1:])])
								if p_L < 30:
									neigh_train.append(curNode)
									neigh_L +=1
									p_L += 1
							elif curNode[0] == "m":
								curNode = random.choice(self.m_p_list_train[int(curNode[1:])])
								if p_L < 30:
									neigh_train.append(curNode)
									neigh_L += 1
									p_L += 1
						else:
							# 这样是返回了
							if i == 0:
								curNode = ('a' + str(j))
							elif i == 1:
								curNode = ('p' + str(j))
							elif i == 2:
								curNode = ('b' + str(j))
							elif i == 3:
								curNode = ('d' + str(j))
							elif i == 4:
								curNode = ('m' + str(j))
							# if curNode == flag_curNode:
							# 	back_count += 1
							# 	if back_count == 2000:
							# 		# print('遇到一个死结，随机游走10万次后',j)
							# 		break

		for i in range(5):
			for j in range(node_n[i]):
				if i == 0:
					a_neigh_list_train[j] = list(a_neigh_list_train[j])
				elif i == 1:
					p_neigh_list_train[j] = list(p_neigh_list_train[j])
				elif i == 2:
					b_neigh_list_train[j] = list(b_neigh_list_train[j])
				elif i == 3:
					d_neigh_list_train[j] = list(d_neigh_list_train[j])
				elif i == 4:
					m_neigh_list_train[j] = list(m_neigh_list_train[j])

		neigh_f = open(self.args.data_path + "het_neigh_train.txt", "w")
		for i in range(5):
			for j in range(node_n[i]):
				if i == 0:
					neigh_train = a_neigh_list_train[j]
					curNode = "a" + str(j)
				elif i == 1:
					neigh_train = p_neigh_list_train[j]
					curNode = "p" + str(j)
				elif i == 2:
					neigh_train = b_neigh_list_train[j]
					curNode = "b" + str(j)
				elif i == 3:
					neigh_train = d_neigh_list_train[j]
					curNode = "d" + str(j)
				elif i == 4:
					neigh_train = m_neigh_list_train[j]
					curNode = "m" + str(j)
				if len(neigh_train):
					neigh_f.write(curNode + ":")
					for k in range(len(neigh_train) - 1):
						neigh_f.write(neigh_train[k] + ",")
					neigh_f.write(neigh_train[-1] + "\n")
		neigh_f.close()

		# 负采样比例计算(). 每一种配对出现的可能性。
	def compute_sample_p(self):
		print("computing sampling ratio for each kind of triple ...")
		window = self.args.window
		walk_L = self.args.walk_L
		A_n = self.args.A_n
		P_n = self.args.P_n
		B_n = self.args.B_n
		D_n = self.args.D_n
		M_n = self.args.M_n

		total_triple_n = [0.0] * 25 # nine kinds of triples，原来是3*3， 现在是4*4, 方法+数据版本是5*5
		# deep walk的结果。 随机游走10遍，每次从author开始，先到paper，再到别的。共有 10*AN行，每行 30个数。
		het_walk_f = open(self.args.data_path + "het_random_walk_test.txt", "r")
		centerNode = ''
		neighNode = ''
		# line 有 10倍 An行。
		for line in het_walk_f:
			line = line.strip()
			path = []
			path_list = re.split(' ', line)
			for i in range(len(path_list)):
				path.append(path_list[i])
				#  30个数。每个走前后 共10个。 大概270个数字。所以总数是270*10*AN。
			for j in range(walk_L):
				centerNode = path[j]
				if len(centerNode) > 1:
					if centerNode[0] == 'a':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a':
									total_triple_n[0] += 1
								elif neighNode[0] == 'p':
									total_triple_n[1] += 1
								elif neighNode[0] == 'b':
									total_triple_n[2] += 1
								elif neighNode[0] == 'd':
									total_triple_n[3] += 1
								elif neighNode[0] == 'm':
									total_triple_n[4] += 1
					elif centerNode[0]=='p':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a':
									total_triple_n[5] += 1
								elif neighNode[0] == 'p':
									total_triple_n[6] += 1
								elif neighNode[0] == 'b':
									total_triple_n[7] += 1
								elif neighNode[0] == 'd':
									total_triple_n[8] += 1
								elif neighNode[0] == 'm':
									total_triple_n[9] += 1
					elif centerNode[0]=='b':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a':
									total_triple_n[10] += 1
								elif neighNode[0] == 'p':
									total_triple_n[11] += 1
								elif neighNode[0] == 'b':
									total_triple_n[12] += 1
								elif neighNode[0] == 'd':
									total_triple_n[13] += 1
								elif neighNode[0] == 'm':
									total_triple_n[14] += 1
					elif centerNode[0] == 'd':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a':
									total_triple_n[15] += 1
								elif neighNode[0] == 'p':
									total_triple_n[16] += 1
								elif neighNode[0] == 'b':
									total_triple_n[17] += 1
								elif neighNode[0] == 'd':
									total_triple_n[18] += 1
								elif neighNode[0] == 'm':
									total_triple_n[19] += 1
					elif centerNode[0] == 'm':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a':
									total_triple_n[20] += 1
								elif neighNode[0] == 'p':
									total_triple_n[21] += 1
								elif neighNode[0] == 'b':
									total_triple_n[22] += 1
								elif neighNode[0] == 'd':
									total_triple_n[23] += 1
								elif neighNode[0] == 'm':
									total_triple_n[24] += 1
		het_walk_f.close()
#此处sample ration 需要加深理解，大概意思：控制每一类的采样数量。数量太大内存爆炸
# total: 270 * 10 * An
# ratio = total_triple_n[i]/total = total_triple_n[i]/ (270*10*An)
		for i in range(len(total_triple_n)):
			total_triple_n[i] = self.args.batch_s / (total_triple_n[i] * 10)
			# 数量越大，采样率越小。是这个数：（2000/total_triple_n[i]），这个数的大小
			# 假设1次给 2000个词，那么就是我们理解的ratio的倒数。
		print("sampling ratio computing finish.")

		return total_triple_n

		# 防止内存爆炸
	def sample_het_walk_triple(self):
		print ("sampling triple relations ...")
		triple_list = [[] for k in range(25)]
		window = self.args.window
		walk_L = self.args.walk_L
		A_n = self.args.A_n
		P_n = self.args.P_n
		B_n = self.args.B_n
		D_n = self.args.D_n
		M_n = self.args.M_n
		# the output of the above function
		triple_sample_p = self.triple_sample_p # use sampling to avoid memory explosion
		het_walk_f = open(self.args.data_path + "het_random_walk_test.txt", "r")
		centerNode = ''
		neighNode = ''
		for line in het_walk_f:
			line = line.strip()
			path = []
			path_list = re.split(' ', line)
			for i in range(len(path_list)):
				path.append(path_list[i])
			for j in range(walk_L):
				centerNode = path[j]
				if len(centerNode) > 1:
					if centerNode[0] == 'a':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								# neighNode是相关节点；即contex，正例。 neg是负例。
								neighNode = path[k]
								# 小于随机数，才进行。
								if neighNode[0] == 'a' and random.random() < triple_sample_p[0]:
									negNode = random.randint(0, A_n - 1)
									# 如果是空的，再抽。
									while len(self.a_p_list_train[negNode]) == 0:
										negNode = random.randint(0, A_n - 1)
									# random negative sampling get similar performance as noise distribution sampling
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[0].append(triple)
								elif neighNode[0] == 'p' and random.random() < triple_sample_p[1]:
									negNode = random.randint(0, P_n - 1)
									while len(self.p_a_list_train[negNode]) == 0:
										negNode = random.randint(0, P_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[1].append(triple)
								elif neighNode[0] == 'b' and random.random() < triple_sample_p[2]:
									negNode = random.randint(0, B_n - 1)
									while len(self.b_p_list_train[negNode]) == 0:
										negNode = random.randint(0, B_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[2].append(triple)
								elif neighNode[0] == 'd' and random.random() < triple_sample_p[3]:
									negNode = random.randint(0, D_n - 1)
									while len(self.d_p_list_train[negNode]) == 0:
										negNode = random.randint(0, D_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[3].append(triple)
								elif neighNode[0] == 'm' and random.random() < triple_sample_p[4]:
									negNode = random.randint(0, M_n - 1)
									while len(self.m_p_list_train[negNode]) == 0:
										negNode = random.randint(0, M_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[4].append(triple)
					elif centerNode[0]=='p':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a' and random.random() < triple_sample_p[5]:
									negNode = random.randint(0, A_n - 1)
									while len(self.a_p_list_train[negNode]) == 0:
										negNode = random.randint(0, A_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[5].append(triple)
								elif neighNode[0] == 'p' and random.random() < triple_sample_p[6]:
									negNode = random.randint(0, P_n - 1)
									while len(self.p_a_list_train[negNode]) == 0:
										negNode = random.randint(0, P_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[6].append(triple)
								elif neighNode[0] == 'b' and random.random() < triple_sample_p[7]:
									negNode = random.randint(0, B_n - 1)
									while len(self.b_p_list_train[negNode]) == 0:
										negNode = random.randint(0, B_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[7].append(triple)
								elif neighNode[0] == 'd' and random.random() < triple_sample_p[8]:
									negNode = random.randint(0, D_n - 1)
									while len(self.d_p_list_train[negNode]) == 0:
										negNode = random.randint(0, D_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[8].append(triple)
								elif neighNode[0] == 'm' and random.random() < triple_sample_p[9]:
									negNode = random.randint(0, M_n - 1)
									while len(self.m_p_list_train[negNode]) == 0:
										negNode = random.randint(0, M_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[9].append(triple)
					elif centerNode[0]=='b':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a' and random.random() < triple_sample_p[10]:
									negNode = random.randint(0, A_n - 1)
									while len(self.a_p_list_train[negNode]) == 0:
										negNode = random.randint(0, A_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[10].append(triple)
								elif neighNode[0] == 'p' and random.random() < triple_sample_p[11]:
									negNode = random.randint(0, P_n - 1)
									while len(self.p_a_list_train[negNode]) == 0:
										negNode = random.randint(0, P_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[11].append(triple)
								elif neighNode[0] == 'b' and random.random() < triple_sample_p[12]:
									negNode = random.randint(0, B_n - 1)
									while len(self.b_p_list_train[negNode]) == 0:
										negNode = random.randint(0, B_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[12].append(triple)
								elif neighNode[0] == 'd' and random.random() < triple_sample_p[13]:
									negNode = random.randint(0, D_n - 1)
									while len(self.d_p_list_train[negNode]) == 0:
										negNode = random.randint(0, D_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[13].append(triple)
								elif neighNode[0] == 'm' and random.random() < triple_sample_p[14]:
									negNode = random.randint(0, M_n - 1)
									while len(self.m_p_list_train[negNode]) == 0:
										negNode = random.randint(0, M_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[14].append(triple)
					elif centerNode[0] == 'd':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a' and random.random() < triple_sample_p[15]:
									negNode = random.randint(0, A_n - 1)
									while len(self.a_p_list_train[negNode]) == 0:
										negNode = random.randint(0, A_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[15].append(triple)
								elif neighNode[0] == 'p' and random.random() < triple_sample_p[16]:
									negNode = random.randint(0, P_n - 1)
									while len(self.p_a_list_train[negNode]) == 0:
										negNode = random.randint(0, P_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[16].append(triple)
								elif neighNode[0] == 'b' and random.random() < triple_sample_p[17]:
									negNode = random.randint(0, B_n - 1)
									while len(self.b_p_list_train[negNode]) == 0:
										negNode = random.randint(0, B_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[17].append(triple)
								elif neighNode[0] == 'd' and random.random() < triple_sample_p[18]:
									negNode = random.randint(0, D_n - 1)
									while len(self.d_p_list_train[negNode]) == 0:
										negNode = random.randint(0, D_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[18].append(triple)
								elif neighNode[0] == 'm' and random.random() < triple_sample_p[19]:
									negNode = random.randint(0, M_n - 1)
									while len(self.m_p_list_train[negNode]) == 0:
										negNode = random.randint(0, M_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[19].append(triple)
					elif centerNode[0] == 'm':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a' and random.random() < triple_sample_p[20]:
									negNode = random.randint(0, A_n - 1)
									while len(self.a_p_list_train[negNode]) == 0:
										negNode = random.randint(0, A_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[20].append(triple)
								elif neighNode[0] == 'p' and random.random() < triple_sample_p[21]:
									negNode = random.randint(0, P_n - 1)
									while len(self.p_a_list_train[negNode]) == 0:
										negNode = random.randint(0, P_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[21].append(triple)
								elif neighNode[0] == 'b' and random.random() < triple_sample_p[22]:
									negNode = random.randint(0, B_n - 1)
									while len(self.b_p_list_train[negNode]) == 0:
										negNode = random.randint(0, B_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[22].append(triple)
								elif neighNode[0] == 'd' and random.random() < triple_sample_p[23]:
									negNode = random.randint(0, D_n - 1)
									while len(self.d_p_list_train[negNode]) == 0:
										negNode = random.randint(0, D_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[23].append(triple)
								elif neighNode[0] == 'm' and random.random() < triple_sample_p[24]:
									negNode = random.randint(0, M_n - 1)
									while len(self.m_p_list_train[negNode]) == 0:
										negNode = random.randint(0, M_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[24].append(triple)
		het_walk_f.close()
			# 这里都是随机游走到的，才会进入triple.的确有个别paper没有author
		return triple_list