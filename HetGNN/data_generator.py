import six.moves.cPickle as pickle
import numpy as np
import string
import re
import random
import math
from collections import Counter
from itertools import *
from tqdm import tqdm

# 我们的测试中，三类节点topk的数值是10,10,10；重启的随机采样为34,33,33

# 这块不用再重复，和 input data process是一致的
class input_data(object):
	def __init__(self, args):
		self.args = args
# 这里，把对应的参数改为我们的数据
		a_p_list_train = [[] for k in range(self.args.A_n)]
		p_a_list_train = [[] for k in range(self.args.P_n)]
		p_p_cite_list_train = [[] for k in range(self.args.P_n)]
		# 这里可能有问题，因为不是每个东西都有citation
		v_p_list_train = [[] for k in range(self.args.V_n)]
		# 我们的pv也是一对多
		p_v = [[] for k in range(self.args.P_n)]

		d_p_list_train = [[] for k in range(self.args.D_n)]
		# 我们的pv也是一对多
		p_d = [[] for k in range(self.args.P_n)]

		# relation_f = ["a_p_list_train.txt", "p_a_list_train.txt",\
		#  "p_p_citation_list.txt", "v_p_list_train.txt"]
		# 暂时没有pp
		relation_f = ["a_p_list_train.txt", "p_a_list_train.txt", "v_p_list_train.txt",
                    'p_v.txt', "d_p_list_train.txt", 'p_d.txt', "p_p_citation_list.txt"]
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
				elif f_name == 'p_v.txt':
					for j in range(len(neigh_list_id)):
						p_v[node_id].append('v'+str(neigh_list_id[j]))
				elif f_name == 'v_p_list_train.txt':
					for j in range(len(neigh_list_id)):
						v_p_list_train[node_id].append('p'+str(neigh_list_id[j]))
				elif f_name == 'p_d.txt':
					for j in range(len(neigh_list_id)):
						p_d[node_id].append('d'+str(neigh_list_id[j]))
				elif f_name == 'd_p_list_train.txt':
					for j in range(len(neigh_list_id)):
						d_p_list_train[node_id].append('p'+str(neigh_list_id[j]))
			neigh_f.close()

		#store paper venue
		# 这个不适用，是一对多
		# p_v = [0] * self.args.P_n
		# p_v_f = open(self.args.data_path + 'p_v.txt', "r")
		# for line in p_v_f:
		# 	line = line.strip()
		# 	p_id = int(re.split(',',line)[0])
		# 	v_id = int(re.split(',',line)[1])
		# 	p_v[p_id] = v_id
		# p_v_f.close()

		#paper neighbor: author + citation + venue 
		#这里我们暂时没有citation，且Venue不止一个
		#补充了citation，但有的会没有；不要紧，没有的话就是空的
		#补充 dataset
		p_neigh_list_train = [[] for k in range(self.args.P_n)]
		for i in range(self.args.P_n):
			p_neigh_list_train[i] += p_a_list_train[i]
			p_neigh_list_train[i] += p_p_cite_list_train[i] 
			p_neigh_list_train[i] += p_v[i]
			p_neigh_list_train[i] += p_d[i]


		self.a_p_list_train =  a_p_list_train
		self.p_a_list_train = p_a_list_train
		self.p_p_cite_list_train = p_p_cite_list_train
		self.p_neigh_list_train = p_neigh_list_train
		self.v_p_list_train = v_p_list_train
		self.d_p_list_train = d_p_list_train

		if self.args.train_test_label != 2:
			self.triple_sample_p = self.compute_sample_p()
			# 这里我们可否直接读进来一个728维度的变量呢，后边需要再细致修改
			# 确定每一块的采样率
			#store paper content pre-trained embedding
			# 这里的ebd注意下,128维度，且第一行是行数+维度
			p_abstract_embed = np.zeros((self.args.P_n, self.args.in_f_d))
			p_a_e_f = open(self.args.data_path + "p_abstract_embed.txt", "r")
			for line in islice(p_a_e_f, 1, None): # 这里跳过了第一行
				values = line.split()
				index = int(values[0])
				embeds = np.asarray(values[1:], dtype='float32')
				p_abstract_embed[index] = embeds
			p_a_e_f.close()

			# p_title_embed = np.zeros((self.args.P_n, self.args.in_f_d))
			# p_t_e_f = open(self.args.data_path + "p_title_embed.txt", "r")
			# for line in islice(p_t_e_f, 1, None):
			# 	values = line.split()
			# 	index = int(values[0])
			# 	embeds = np.asarray(values[1:], dtype='float32')
			# 	p_title_embed[index] = embeds
			# p_t_e_f.close()

			self.p_abstract_embed = p_abstract_embed
			# self.p_abstract_embed = ''
			# self.p_title_embed = p_title_embed
			self.p_title_embed = ''

			#store pre-trained network/content embedding
			# 这块可以有，注意V要变B
			a_net_embed = np.zeros((self.args.A_n, self.args.in_f_d))
			p_net_embed = np.zeros((self.args.P_n, self.args.in_f_d))
			# 注意维度
			v_net_embed = np.zeros((self.args.V_n, self.args.in_f_d))
			d_net_embed = np.zeros((self.args.D_n, self.args.in_f_d))
			# 注意路径，这个net是Word2Vec得到的，待会儿要生成一下
			net_e_f = open(self.args.data_path + "node_net_embedding.txt", "r")
			for line in islice(net_e_f, 1, None):
				line = line.strip()
				index = re.split(' ', line)[0]
				if len(index) and (index[0] == 'a' or index[0] == 'v' or index[0] == 'p' or index[0] == 'd'):
					embeds = np.asarray(re.split(' ', line)[1:], dtype='float32')
					if index[0] == 'a':
						a_net_embed[int(index[1:])] = embeds
					elif index[0] == 'v':
						v_net_embed[int(index[1:])] = embeds
					elif index[0] == 'p':
						p_net_embed[int(index[1:])] = embeds
					elif index[0] == 'd':
						d_net_embed[int(index[1:])] = embeds
			net_e_f.close()
			# pv embedding； 我们的pv embedding应该和下面的pa是类似的，因为是一对多
			# p_v_net_embed = np.zeros((self.args.P_n, self.args.in_f_d))
			# p_v = [0] * self.args.P_n
			# p_v_f = open(self.args.data_path + "p_v.txt", "r")
			# for line in p_v_f:
			# 	line = line.strip()
			# 	p_id = int(re.split(',', line)[0])
			# 	v_id = int(re.split(',', line)[1])
			# 	p_v[p_id] = v_id
			# 	p_v_net_embed[p_id] = v_net_embed[v_id]
			# p_v_f.close()
			p_v_net_embed = np.zeros((self.args.P_n, self.args.in_f_d))
			for i in range(self.args.P_n):
				# 这里可以对上，原因是p_v的长度是按照paper的总量来的，有些可能是空的
				if len(p_v[i]):
					for j in range(len(p_v[i])):
						v_id = int(p_v[i][j][1:])
						p_v_net_embed[i] = np.add(p_v_net_embed[i], v_net_embed[v_id])
					p_v_net_embed[i] = p_v_net_embed[i] / len(p_v[i])

			p_d_net_embed = np.zeros((self.args.P_n, self.args.in_f_d))
			for i in range(self.args.P_n):
				# 这里可以对上，原因是p_d的长度是按照paper的总量来的，有些可能是空的
				if len(p_d[i]):
					for j in range(len(p_d[i])):
						d_id = int(p_d[i][j][1:])
						p_d_net_embed[i] = np.add(p_d_net_embed[i], d_net_embed[d_id])
					p_d_net_embed[i] = p_d_net_embed[i] / len(p_d[i])

			p_a_net_embed = np.zeros((self.args.P_n, self.args.in_f_d))
			for i in range(self.args.P_n):
				# 这块的p能对上
				if len(p_a_list_train[i]):
					for j in range(len(p_a_list_train[i])):
						a_id = int(p_a_list_train[i][j][1:])
						p_a_net_embed[i] = np.add(p_a_net_embed[i], a_net_embed[a_id])
					p_a_net_embed[i] = p_a_net_embed[i] / len(p_a_list_train[i])
				# 参考文献的embedding
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
			# #empirically use 5 paper embedding for venue content embeding generation
			v_text_embed = np.zeros((self.args.V_n, self.args.in_f_d * 5))
			for i in range(self.args.V_n):
				if len(v_p_list_train[i]):
					feature_temp = []
					if len(v_p_list_train[i]) >= 5:
						# id_list_temp = random.sample(a_p_list_train[i], 5)
						for j in range(5):
							feature_temp.append(p_abstract_embed[int(v_p_list_train[i][j][1:])])
					else:
						for j in range(len(v_p_list_train[i])):
							feature_temp.append(p_abstract_embed[int(v_p_list_train[i][j][1:])])
						for k in range(len(v_p_list_train[i]), 5):
							feature_temp.append(p_abstract_embed[int(v_p_list_train[i][-1][1:])])

					feature_temp = np.reshape(np.asarray(feature_temp), [1, -1])
					v_text_embed[i] = feature_temp

			self.p_v = p_v
			self.p_d = p_d
			self.p_v_net_embed = p_v_net_embed
			self.p_d_net_embed = p_d_net_embed
			self.p_a_net_embed = p_a_net_embed
			self.p_ref_net_embed = p_ref_net_embed
			self.p_net_embed = p_net_embed
			self.a_net_embed = a_net_embed
			self.a_text_embed = a_text_embed
			self.v_net_embed = v_net_embed
			self.v_text_embed = v_text_embed
			self.d_net_embed = d_net_embed
			self.d_text_embed = d_text_embed
			











			#store neighbor set from random walk sequence 
			a_neigh_list_train = [[[] for i in range(self.args.A_n)] for j in range(4)]
			p_neigh_list_train = [[[] for i in range(self.args.P_n)] for j in range(4)]
			v_neigh_list_train = [[[] for i in range(self.args.V_n)] for j in range(4)]
			d_neigh_list_train = [[[] for i in range(self.args.D_n)] for j in range(4)]
			# 这块要随机游走先得到，这里待会儿必须先跑随机游走，需要het_neigh_train.txt；这里的采样数量（10，10，3）和后面的a_L, v_L以及p_L
			# 这里是重启随机游走已经采样完了，从这里各类型存储top个节点。每个节点找了100个邻居，在这一步从里面采样23个（10+10+3），我们采样30个，10，10，10
			het_neigh_train_f = open(self.args.data_path + "het_neigh_train.txt", "r")
			for line in het_neigh_train_f:
				line = line.strip()
				node_id = re.split(':', line)[0]
				neigh = re.split(':', line)[1]
				neigh_list = re.split(',', neigh)
				# 只要是随机游走到了，都会放进来；没放进来的，都会被空置，就会出问题
				if node_id[0] == 'a' and len(node_id) > 1:
					for j in range(len(neigh_list)):
						# 不管跟啥连着，都放在这里；所以有的可能没有
						if neigh_list[j][0] == 'a':
							a_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'p':
							a_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'v':
							a_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'd':
							a_neigh_list_train[3][int(node_id[1:])].append(int(neigh_list[j][1:]))
				elif node_id[0] == 'p' and len(node_id) > 1:
					for j in range(len(neigh_list)):
						if neigh_list[j][0] == 'a':
							p_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'p':
							p_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'v':
							p_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'd':
							p_neigh_list_train[3][int(node_id[1:])].append(int(neigh_list[j][1:]))
				elif node_id[0] == 'v' and len(node_id) > 1:
					for j in range(len(neigh_list)):
						if neigh_list[j][0] == 'a':
							v_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'p':
							v_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'v':
							v_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'd':
							v_neigh_list_train[3][int(node_id[1:])].append(int(neigh_list[j][1:]))
				elif node_id[0] == 'd' and len(node_id) > 1:
					for j in range(len(neigh_list)):
						if neigh_list[j][0] == 'a':
							d_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'p':
							d_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'v':
							d_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'd':
							d_neigh_list_train[3][int(node_id[1:])].append(int(neigh_list[j][1:]))
			het_neigh_train_f.close()
			#print a_neigh_list_train[0][1]

			#store top neighbor set (based on frequency) from random walk sequence  抽样，这三个的结构一样，每个节点，3类邻居。
			a_neigh_list_train_top = [[[] for i in range(self.args.A_n)] for j in range(4)]
			p_neigh_list_train_top = [[[] for i in range(self.args.P_n)] for j in range(4)]
			v_neigh_list_train_top = [[[] for i in range(self.args.V_n)] for j in range(4)]
			d_neigh_list_train_top = [[[] for i in range(self.args.D_n)] for j in range(4)]
			# 为什么是10 10 3？这里必须要重新考虑！！！
			top_k = [10, 10, 10, 5] #fix each neighor type size
			for i in range(self.args.A_n):
				for j in range(4):
					a_neigh_list_train_temp = Counter(a_neigh_list_train[j][i])
					top_list = a_neigh_list_train_temp.most_common(top_k[j])
					neigh_size = 0
					if j == 3:
						neigh_size = 5
						# 为什么当j到第三个时，neigh_size缩小为3？
					else:
						neigh_size = 10
					for k in range(len(top_list)):
						a_neigh_list_train_top[j][i].append(int(top_list[k][0]))
					if len(a_neigh_list_train_top[j][i]) and len(a_neigh_list_train_top[j][i]) < neigh_size:
						for l in range(len(a_neigh_list_train_top[j][i]), neigh_size):
							a_neigh_list_train_top[j][i].append(random.choice(a_neigh_list_train_top[j][i]))

			for i in range(self.args.P_n):
				for j in range(4):
					p_neigh_list_train_temp = Counter(p_neigh_list_train[j][i])
					top_list = p_neigh_list_train_temp.most_common(top_k[j])
					neigh_size = 0
					if j == 3:
						neigh_size = 5
					else:
						neigh_size = 10
					for k in range(len(top_list)):
						p_neigh_list_train_top[j][i].append(int(top_list[k][0]))
					if len(p_neigh_list_train_top[j][i]) and len(p_neigh_list_train_top[j][i]) < neigh_size:
						for l in range(len(p_neigh_list_train_top[j][i]), neigh_size):
							p_neigh_list_train_top[j][i].append(random.choice(p_neigh_list_train_top[j][i]))

			for i in range(self.args.V_n):
				for j in range(4):
					v_neigh_list_train_temp = Counter(v_neigh_list_train[j][i])
					top_list = v_neigh_list_train_temp.most_common(top_k[j])
					neigh_size = 0
					if j == 3:
						neigh_size = 5
					else:
						neigh_size = 10
					for k in range(len(top_list)):
						v_neigh_list_train_top[j][i].append(int(top_list[k][0]))
					if len(v_neigh_list_train_top[j][i]) and len(v_neigh_list_train_top[j][i]) < neigh_size:
						for l in range(len(v_neigh_list_train_top[j][i]), neigh_size):
							v_neigh_list_train_top[j][i].append(random.choice(v_neigh_list_train_top[j][i]))
			
			for i in range(self.args.D_n):
				for j in range(4):
					d_neigh_list_train_temp = Counter(d_neigh_list_train[j][i])
					top_list = d_neigh_list_train_temp.most_common(top_k[j])
					neigh_size = 0
					if j == 3:
						neigh_size = 5
					else:
						neigh_size = 10
					for k in range(len(top_list)):
						d_neigh_list_train_top[j][i].append(int(top_list[k][0]))
					if len(d_neigh_list_train_top[j][i]) and len(d_neigh_list_train_top[j][i]) < neigh_size:
						for l in range(len(d_neigh_list_train_top[j][i]), neigh_size):
							d_neigh_list_train_top[j][i].append(
								random.choice(d_neigh_list_train_top[j][i]))

			a_neigh_list_train[:] = []
			p_neigh_list_train[:] = []
			v_neigh_list_train[:] = []
			d_neigh_list_train[:] = []

			self.a_neigh_list_train = a_neigh_list_train_top
			self.p_neigh_list_train = p_neigh_list_train_top
			self.v_neigh_list_train = v_neigh_list_train_top
			self.d_neigh_list_train = d_neigh_list_train_top






















			#store ids of author/paper/venue used in training
			train_id_list = [[] for i in range(4)]
			for i in range(4):
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
					for l in range(self.args.V_n):
						if len(v_neigh_list_train_top[i][l]):
							train_id_list[i].append(l)
					self.v_train_id_list = np.array(train_id_list[i])
				elif i == 3:
					for l in range(self.args.D_n):
						if len(d_neigh_list_train_top[i][l]):
							train_id_list[i].append(l)
					self.d_train_id_list = np.array(train_id_list[i])
			#print (len(self.v_train_id_list))		

			#重启随机游走
	def het_walk_restart(self):
		a_neigh_list_train = [[] for k in range(self.args.A_n)]
		p_neigh_list_train = [[] for k in range(self.args.P_n)]
		v_neigh_list_train = [[] for k in range(self.args.V_n)]
		d_neigh_list_train = [[] for k in range(self.args.D_n)]
		#generate neighbor set via random walk with restart
		node_n = [self.args.A_n, self.args.P_n, self.args.V_n, self.args.D_n]
		for i in range(4):
			for j in range(node_n[i]):
				if i == 0:
					# temp是从第一个节点开始抽
					neigh_temp = self.a_p_list_train[j]
					# train是选一个对应位置的空白列表，往里放东西
					neigh_train = a_neigh_list_train[j]
					curNode = "a" + str(j)
				elif i == 1:
					neigh_temp = self.p_a_list_train[j]
					neigh_train = p_neigh_list_train[j]
					curNode = "p" + str(j)
				elif i == 2:
					# 每个v都有p对应
					neigh_temp = self.v_p_list_train[j]
					neigh_train = v_neigh_list_train[j]
					curNode = "v" + str(j)
				elif i == 3:
					# 每个v都有p对应
					neigh_temp = self.d_p_list_train[j]
					neigh_train = d_neigh_list_train[j]
					curNode = "d" + str(j)
					# 此处表示该类型的节点是存在的，才会在后续考虑
				if len(neigh_temp):
					# flag_curNode = curNode
					# [删除了这些条件，因为我们有dataset，有底气] 加一些限制条件，假如返回最初的起点超过10次，说明大概率找不到100个节点了，就continue
					# back_count = 0
					neigh_L = 0
					a_L = 0
					p_L = 0
					v_L = 0
					d_L = 0
					while neigh_L < 100: #maximum neighbor size = 100, 28,28,28,16
						rand_p = random.random() #return p； 返回的概率
						if rand_p > 0.5:
							if curNode[0] == "a":
								curNode = random.choice(self.a_p_list_train[int(curNode[1:])])
								if p_L < 29: #size constraint (make sure each type of neighobr is sampled)
									neigh_train.append(curNode)
									neigh_L += 1
									p_L += 1
							elif curNode[0] == "p":
								curNode = random.choice(self.p_neigh_list_train[int(curNode[1:])])
								if curNode != ('a' + str(j)) and curNode[0] == 'a' and a_L < 29:
									neigh_train.append(curNode)
									neigh_L += 1
									a_L += 1
								elif curNode[0] == 'v':
									if v_L < 29:
										neigh_train.append(curNode)
										neigh_L += 1
										v_L += 1
								elif curNode[0] == 'd':
									if d_L < 17:
										neigh_train.append(curNode)
										neigh_L += 1
										d_L += 1
							elif curNode[0] == "v":
								curNode = random.choice(self.v_p_list_train[int(curNode[1:])])
								if p_L < 29:
									neigh_train.append(curNode)
									neigh_L +=1
									p_L += 1
							elif curNode[0] == "d":
								curNode = random.choice(self.d_p_list_train[int(curNode[1:])])
								if p_L < 29:
									neigh_train.append(curNode)
									neigh_L +=1
									p_L += 1
						else:
							# 这样是返回了
							if i == 0:
								curNode = ('a' + str(j))
							elif i == 1:
								curNode = ('p' + str(j))
							elif i == 2:
								curNode = ('v' + str(j))
							elif i == 3:
								curNode = ('d' + str(j))
							# if curNode == flag_curNode:
							# 	back_count += 1
							# 	if back_count == 2000:
							# 		# 说明死循环了，源代码中的v是会议，不存在这种情况【原因：只存在一个作者节点，根本凑不够35个】
							# 		# print('遇到一个死结，随机游走10万次后',j)
							# 		break

		for i in range(4):
			for j in range(node_n[i]):
				if i == 0:
					a_neigh_list_train[j] = list(a_neigh_list_train[j])
				elif i == 1:
					p_neigh_list_train[j] = list(p_neigh_list_train[j])
				elif i == 2:
					v_neigh_list_train[j] = list(v_neigh_list_train[j])
				elif i == 3:
					d_neigh_list_train[j] = list(d_neigh_list_train[j])

		neigh_f = open(self.args.data_path + "het_neigh_train.txt", "w")
		for i in range(4):
			for j in range(node_n[i]):
				if i == 0:
					neigh_train = a_neigh_list_train[j]
					curNode = "a" + str(j)
				elif i == 1:
					neigh_train = p_neigh_list_train[j]
					curNode = "p" + str(j)
				elif i == 2:
					neigh_train = v_neigh_list_train[j]
					curNode = "v" + str(j)
					# 这个东西不能空，有空的话就没有被游走到。
				elif i == 3:
					neigh_train = d_neigh_list_train[j]
					curNode = "d" + str(j)
				if len(neigh_train):
					neigh_f.write(curNode + ":")
					for k in range(len(neigh_train) - 1):
						neigh_f.write(neigh_train[k] + ",")
					neigh_f.write(neigh_train[-1] + "\n")
		neigh_f.close()

		# 负采样比例计算
	def compute_sample_p(self):
		print("computing sampling ratio for each kind of triple ...")
		window = self.args.window
		walk_L = self.args.walk_L
		A_n = self.args.A_n
		P_n = self.args.P_n
		V_n = self.args.V_n
		D_n = self.args.D_n

		total_triple_n = [0.0] * 16 # nine kinds of triples，原来是3*3， 现在是4*4
		# 这个东西，注意可能有某个作者没有V，但都有paper
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
								neighNode = path[k]
								if neighNode[0] == 'a':
									total_triple_n[0] += 1
								elif neighNode[0] == 'p':
									total_triple_n[1] += 1
								elif neighNode[0] == 'v':
									total_triple_n[2] += 1
								elif neighNode[0] == 'd':
									total_triple_n[3] += 1
					elif centerNode[0]=='p':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a':
									total_triple_n[4] += 1
								elif neighNode[0] == 'p':
									total_triple_n[5] += 1
								elif neighNode[0] == 'v':
									total_triple_n[6] += 1
								elif neighNode[0] == 'd':
									total_triple_n[7] += 1
					elif centerNode[0]=='v':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a':
									total_triple_n[8] += 1
								elif neighNode[0] == 'p':
									total_triple_n[9] += 1
								elif neighNode[0] == 'v':
									total_triple_n[10] += 1
								elif neighNode[0] == 'd':
									total_triple_n[11] += 1
					elif centerNode[0] == 'd':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a':
									total_triple_n[12] += 1
								elif neighNode[0] == 'p':
									total_triple_n[13] += 1
								elif neighNode[0] == 'v':
									total_triple_n[14] += 1
								elif neighNode[0] == 'd':
									total_triple_n[15] += 1
		het_walk_f.close()

		for i in range(len(total_triple_n)):
			total_triple_n[i] = self.args.batch_s / (total_triple_n[i] * 10)
		print("sampling ratio computing finish.")

		return total_triple_n

		# 防止内存爆炸
	def sample_het_walk_triple(self):
		print ("sampling triple relations ...")
		triple_list = [[] for k in range(16)]
		window = self.args.window
		walk_L = self.args.walk_L
		A_n = self.args.A_n
		P_n = self.args.P_n
		V_n = self.args.V_n
		D_n = self.args.D_n
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
								neighNode = path[k]
								if neighNode[0] == 'a' and random.random() < triple_sample_p[0]:
									negNode = random.randint(0, A_n - 1)
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
								elif neighNode[0] == 'v' and random.random() < triple_sample_p[2]:
									negNode = random.randint(0, V_n - 1)
									while len(self.v_p_list_train[negNode]) == 0:
										negNode = random.randint(0, V_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[2].append(triple)
								elif neighNode[0] == 'd' and random.random() < triple_sample_p[3]:
									negNode = random.randint(0, D_n - 1)
									while len(self.d_p_list_train[negNode]) == 0:
										negNode = random.randint(0, D_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[3].append(triple)
					elif centerNode[0]=='p':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a' and random.random() < triple_sample_p[4]:
									negNode = random.randint(0, A_n - 1)
									while len(self.a_p_list_train[negNode]) == 0:
										negNode = random.randint(0, A_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[4].append(triple)
								elif neighNode[0] == 'p' and random.random() < triple_sample_p[5]:
									negNode = random.randint(0, P_n - 1)
									while len(self.p_a_list_train[negNode]) == 0:
										negNode = random.randint(0, P_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[5].append(triple)
								elif neighNode[0] == 'v' and random.random() < triple_sample_p[6]:
									negNode = random.randint(0, V_n - 1)
									while len(self.v_p_list_train[negNode]) == 0:
										negNode = random.randint(0, V_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[6].append(triple)
								elif neighNode[0] == 'd' and random.random() < triple_sample_p[7]:
									negNode = random.randint(0, D_n - 1)
									while len(self.d_p_list_train[negNode]) == 0:
										negNode = random.randint(0, D_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[7].append(triple)
					elif centerNode[0]=='v':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a' and random.random() < triple_sample_p[8]:
									negNode = random.randint(0, A_n - 1)
									while len(self.a_p_list_train[negNode]) == 0:
										negNode = random.randint(0, A_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[8].append(triple)
								elif neighNode[0] == 'p' and random.random() < triple_sample_p[9]:
									negNode = random.randint(0, P_n - 1)
									while len(self.p_a_list_train[negNode]) == 0:
										negNode = random.randint(0, P_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[9].append(triple)
								elif neighNode[0] == 'v' and random.random() < triple_sample_p[10]:
									negNode = random.randint(0, V_n - 1)
									while len(self.v_p_list_train[negNode]) == 0:
										negNode = random.randint(0, V_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[10].append(triple)
								elif neighNode[0] == 'd' and random.random() < triple_sample_p[11]:
									negNode = random.randint(0, D_n - 1)
									while len(self.d_p_list_train[negNode]) == 0:
										negNode = random.randint(0, D_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[11].append(triple)
					elif centerNode[0] == 'd':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a' and random.random() < triple_sample_p[12]:
									negNode = random.randint(0, A_n - 1)
									while len(self.a_p_list_train[negNode]) == 0:
										negNode = random.randint(0, A_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[12].append(triple)
								elif neighNode[0] == 'p' and random.random() < triple_sample_p[13]:
									negNode = random.randint(0, P_n - 1)
									while len(self.p_a_list_train[negNode]) == 0:
										negNode = random.randint(0, P_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[13].append(triple)
								elif neighNode[0] == 'v' and random.random() < triple_sample_p[14]:
									negNode = random.randint(0, V_n - 1)
									while len(self.v_p_list_train[negNode]) == 0:
										negNode = random.randint(0, V_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[14].append(triple)
								elif neighNode[0] == 'd' and random.random() < triple_sample_p[15]:
									negNode = random.randint(0, D_n - 1)
									while len(self.d_p_list_train[negNode]) == 0:
										negNode = random.randint(0, D_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[15].append(triple)
		het_walk_f.close()
			# 这里都是随机游走到的，才会进入triple.的确有个别paper没有author
		return triple_list


# input_data_class = input_data(args = args)

# input_data_class.het_walk_restart()