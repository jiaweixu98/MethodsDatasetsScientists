# nohup python -u input_data_process.py > input_data_process.log 2>&1 &
import argparse
import re
import random
from itertools import *
from tqdm import tqdm

parser = argparse.ArgumentParser(description = 'input data process')
parser.add_argument('--A_n', type=int, default= 6824,
				   help = 'number of author node')
parser.add_argument('--P_n', type = int, default = 1143,
				   help = 'number of paper node')
parser.add_argument('--B_n', type=int, default=8633,
				   help = 'number of bio node')
parser.add_argument('--D_n', type=int, default = 128,
                     help='number of dataset node')
parser.add_argument('--M_n', type=int, default=89,
                     help='number of method node')
parser.add_argument('--data_path', type=str, default= '../../../data/subsetHetGNNdata/',
                    help='path to data')
parser.add_argument('--walk_n', type = int, default = 10,
			   help='number of walk per root node')
parser.add_argument('--walk_L', type = int, default = 30,
			   help='length of each walk')
parser.add_argument('--window', type = int, default = 7,
			   help='window size for relation extration')
parser.add_argument('--T_split', type = int, default = 2012,
			   help = 'split time of train/test data')

args = parser.parse_args()
print(args)


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

		relation_f = ["a_p_list_train.txt", "p_a_list_train.txt", "p_p_citation_list.txt",
                    "b_p_list_train.txt", 'p_b.txt', "d_p_list_train.txt", 'p_d.txt', "m_p_list_train.txt", 'p_m.txt', ]
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
						p_m[node_id].append('m'+str(neigh_list_id[j]))
				elif f_name == 'm_p_list_train.txt':
					for j in range(len(neigh_list_id)):
						m_p_list_train[node_id].append('p'+str(neigh_list_id[j]))
			neigh_f.close()

		#paper neighbor: author + citation + bio + dataset + method
		p_neigh_list_train = [[] for k in range(self.args.P_n)]
		for i in range(self.args.P_n):
			p_neigh_list_train[i] += p_a_list_train[i]
			p_neigh_list_train[i] += p_p_cite_list_train[i]
			p_neigh_list_train[i] += p_b[i]
			p_neigh_list_train[i] += p_d[i]
			p_neigh_list_train[i] += p_m[i]

		self.a_p_list_train = a_p_list_train
		self.p_a_list_train = p_a_list_train
		self.p_p_cite_list_train = p_p_cite_list_train
		self.p_neigh_list_train = p_neigh_list_train
		self.b_p_list_train = b_p_list_train
		self.d_p_list_train = d_p_list_train
		self.m_p_list_train = m_p_list_train

# 下面这三个函数，最重要的是随机游走；其他的是干什么呢？
# 游走，也不用干别的
	def gen_het_rand_walk(self):
		het_walk_f = open(self.args.data_path + "het_random_walk_test.txt", "w")
		for i in tqdm(range(self.args.walk_n)):
			for j in range(self.args.A_n):
				if len(self.a_p_list_train[j]):
					curNode = "a" + str(j)
					het_walk_f.write(curNode + " ")
					for l in range(self.args.walk_L - 1):
						if curNode[0] == "a":
							curNode = int(curNode[1:])
							curNode = random.choice(self.a_p_list_train[curNode])
							het_walk_f.write(curNode + " ")
						elif curNode[0] == "p":
							curNode = int(curNode[1:])
							curNode = random.choice(self.p_neigh_list_train[curNode])
							het_walk_f.write(curNode + " ")
						elif curNode[0] == "b": 
							curNode = int(curNode[1:])
							curNode = random.choice(self.b_p_list_train[curNode])
							het_walk_f.write(curNode + " ")
						elif curNode[0] == "d": 
							curNode = int(curNode[1:])
							curNode = random.choice(self.d_p_list_train[curNode])
							het_walk_f.write(curNode + " ")
						elif curNode[0] == "m": 
							curNode = int(curNode[1:])
							curNode = random.choice(self.m_p_list_train[curNode])
							het_walk_f.write(curNode + " ")
					het_walk_f.write("\n")
		het_walk_f.close()

# 这个函数好像没用
	def gen_meta_rand_walk_APVPA(self):
		meta_walk_f = open(self.args.data_path + "meta_random_walk_APVPA_test.txt", "w")
		#print len(self.p_neigh_list_train)
		for i in range(self.args.walk_n):
			for j in range(self.args.A_n):
				if len(self.a_p_list_train[j]):
					curNode = "a" + str(j)
					preNode = "a" + str(j)
					meta_walk_f.write(curNode + " ")
					for l in range(self.args.walk_L - 1):
						if curNode[0] == "a":
							preNode = curNode
							curNode = int(curNode[1:])
							curNode = random.choice(self.a_p_list_train[curNode])
							meta_walk_f.write(curNode + " ")
						elif curNode[0] == "p":
							curNode = int(curNode[1:])
							if preNode[0] == "a":
								preNode = "p" + str(curNode)
								curNode = "p" + str(self.p_v[curNode])
								meta_walk_f.write(curNode + " ")
							else:
								preNode = "p" + str(curNode)
								curNode = random.choice(self.p_neigh_list_train[curNode])
								meta_walk_f.write(curNode + " ")
						elif curNode[0] == "v": 
							preNode = curNode
							curNode = int(curNode[1:])
							curNode = random.choice(self.v_p_list_train[curNode])
							meta_walk_f.write(curNode + " ")
					meta_walk_f.write("\n")
		meta_walk_f.close()


#生成data，不一定有用
	def a_a_collaborate_train_test(self):
		a_a_list_train = [[] for k in range(self.args.A_n)]
		a_a_list_test = [[] for k in range(self.args.A_n)]
		p_a_list = [self.p_a_list_train, self.p_a_list_test]
		
		for t in range(len(p_a_list)):
			for i in range(len(p_a_list[t])):
				for j in range(len(p_a_list[t][i])):
					for k in range(j+1, len(p_a_list[t][i])):
						if t == 0:
							a_a_list_train[int(p_a_list[t][i][j][1:])].append(int(p_a_list[t][i][k][1:]))
							a_a_list_train[int(p_a_list[t][i][k][1:])].append(int(p_a_list[t][i][j][1:]))
						else:#remove duplication in test and only consider existing authors
							if len(a_a_list_train[int(p_a_list[t][i][j][1:])]) and len(a_a_list_train[int(p_a_list[t][i][k][1:])]):#transductive case
								if int(p_a_list[t][i][k][1:]) not in a_a_list_train[int(p_a_list[t][i][j][1:])]:
									a_a_list_test[int(p_a_list[t][i][j][1:])].append(int(p_a_list[t][i][k][1:]))
								if int(p_a_list[t][i][j][1:]) not in a_a_list_train[int(p_a_list[t][i][k][1:])]:
									a_a_list_test[int(p_a_list[t][i][k][1:])].append(int(p_a_list[t][i][j][1:]))
		
		#print (a_a_list_train[1])

		for i in range(self.args.A_n):
			a_a_list_train[i]=list(set(a_a_list_train[i]))
			a_a_list_test[i]=list(set(a_a_list_test[i]))

		a_a_list_train_f = open(args.data_path + "a_a_list_train.txt", "w")
		a_a_list_test_f = open(args.data_path + "a_a_list_test.txt", "w")
		a_a_list = [a_a_list_train, a_a_list_test]
		train_num = 0
		test_num = 0
		for t in range(len(a_a_list)):
			for i in range(len(a_a_list[t])):
				#print (i)
				if len(a_a_list[t][i]):
					if t == 0:
						for j in range(len(a_a_list[t][i])):
							a_a_list_train_f.write("%d, %d, %d\n"%(i, a_a_list[t][i][j], 1))
							node_n = random.randint(0, self.args.A_n - 1)
							while node_n in a_a_list[t][i]: 
								node_n = random.randint(0, self.args.A_n - 1)
							a_a_list_train_f.write("%d, %d, %d\n"%(i, node_n, 0))
							train_num += 2
					else:
						for j in range(len(a_a_list[t][i])):
							a_a_list_test_f.write("%d, %d, %d\n"%(i, a_a_list[t][i][j], 1))
							node_n = random.randint(0, self.args.A_n - 1)
							while node_n in a_a_list[t][i] or node_n in a_a_list_train[i] or len(a_a_list_train[i]) == 0:
								node_n = random.randint(0, self.args.A_n - 1)
							a_a_list_test_f.write("%d, %d, %d\n"%(i, node_n, 0))	 
							test_num += 2
		a_a_list_train_f.close()
		a_a_list_test_f.close()

		print("a_a_train_num: " + str(train_num))
		print("a_a_test_num: " + str(test_num))

#生成data，不一定有用
	def a_p_citation_train_test(self):
		p_time = [0] * args.P_n
		p_time_f = open(args.data_path + "p_time.txt", "r")
		for line in p_time_f:
			line = line.strip()
			p_id = int(re.split('\t',line)[0])
			time = int(re.split('\t',line)[1])
			p_time[p_id] = time + 2005
		p_time_f.close()

		a_p_cite_list_train = [[] for k in range(self.args.A_n)]
		a_p_cite_list_test = [[] for k in range(self.args.A_n)]
		a_p_list = [self.a_p_list_train, self.a_p_list_test]
		p_p_cite_list_train = self.p_p_cite_list_train
		p_p_cite_list_test = self.p_p_cite_list_test
		
		for t in range(len(a_p_list)):
			for i in range(len(a_p_list[t])):
				for j in range(len(a_p_list[t][i])):
					if t == 0:
						p_id = int(a_p_list[t][i][j][1:])
						for k in range(len(p_p_cite_list_train[p_id])):
							a_p_cite_list_train[i].append(int(p_p_cite_list_train[p_id][k][1:]))
					else:#remove duplication in test and only consider existing papers
						if len(self.a_p_list_train[i]):#tranductive inference
							p_id = int(a_p_list[t][i][j][1:])
							for k in range(len(p_p_cite_list_test[p_id])):
								cite_index = int(p_p_cite_list_test[p_id][k][1:])
								if p_time[cite_index] < args.T_split and (cite_index not in a_p_cite_list_train[i]):
									a_p_cite_list_test[i].append(cite_index)


		for i in range(self.args.A_n):
			a_p_cite_list_train[i] = list(set(a_p_cite_list_train[i]))
			a_p_cite_list_test[i] = list(set(a_p_cite_list_test[i]))

		test_count = 0 
		#print (a_p_cite_list_test[56])
		a_p_cite_list_train_f = open(args.data_path + "a_p_cite_list_train.txt", "w")
		a_p_cite_list_test_f = open(args.data_path + "a_p_cite_list_test.txt", "w")
		a_p_cite_list = [a_p_cite_list_train, a_p_cite_list_test]
		train_num = 0
		test_num = 0
		for t in range(len(a_p_cite_list)):
			for i in range(len(a_p_cite_list[t])):
				#print (i)
				#if len(a_p_cite_list[t][i]):
				if t == 0:
					for j in range(len(a_p_cite_list[t][i])):
						a_p_cite_list_train_f.write("%d, %d, %d\n"%(i, a_p_cite_list[t][i][j], 1))
						node_n = random.randint(0, self.args.P_n - 1)
						while node_n in a_p_cite_list[t][i] or node_n in a_p_cite_list_train[i]: 
							node_n = random.randint(0, self.args.P_n - 1)
						a_p_cite_list_train_f.write("%d, %d, %d\n"%(i, node_n, 0))
						train_num += 2
				else:
					for j in range(len(a_p_cite_list[t][i])):
						a_p_cite_list_test_f.write("%d, %d, %d\n"%(i, a_p_cite_list[t][i][j], 1))
						node_n = random.randint(0, self.args.P_n - 1)
						while node_n in a_p_cite_list[t][i] or node_n in a_p_cite_list_train[i]:
							node_n = random.randint(0, self.args.P_n - 1)
						a_p_cite_list_test_f.write("%d, %d, %d\n"%(i, node_n, 0))	 
						test_num += 2
		a_p_cite_list_train_f.close()
		a_p_cite_list_test_f.close()

		print("a_p_cite_train_num: " + str(train_num))
		print("a_p_cite_test_num: " + str(test_num))

#生成data，不一定有用
	def a_v_train_test(self):
		a_v_list_train = [[] for k in range(self.args.A_n)]
		a_v_list_test = [[] for k in range(self.args.A_n)]
		a_p_list = [self.a_p_list_train, self.a_p_list_test]
		for t in range(len(a_p_list)):
			for i in range(len(a_p_list[t])):
				for j in range(len(a_p_list[t][i])):
					p_id = int(a_p_list[t][i][j][1:])
					if t == 0:
						a_v_list_train[i].append(self.p_v[p_id])
					else:
						if self.p_v[p_id] not in a_v_list_train[i] and len(a_v_list_train[i]):
							a_v_list_test[i].append(self.p_v[p_id])

		for k in range(self.args.A_n):
			a_v_list_train[k] = list(set(a_v_list_train[k]))
			a_v_list_test[k] = list(set(a_v_list_test[k]))

		a_v_list_train_f = open(args.data_path + "a_v_list_train.txt", "w")
		a_v_list_test_f = open(args.data_path + "a_v_list_test.txt", "w")
		a_v_list = [a_v_list_train, a_v_list_test]
		# train_num = 0
		# test_num = 0
		# test_a_num = 0
		for t in range(len(a_v_list)):
			for i in range(len(a_v_list[t])):
				if t == 0:
					if len(a_v_list[t][i]):
						a_v_list_train_f.write(str(i)+":")
						for j in range(len(a_v_list[t][i])):
							a_v_list_train_f.write(str(a_v_list[t][i][j])+",")
							#train_num += 1
						a_v_list_train_f.write("\n")
				else:
					if len(a_v_list[t][i]):
						#test_a_num += 1
						a_v_list_test_f.write(str(i)+":")
						for j in range(len(a_v_list[t][i])):
							a_v_list_test_f.write(str(a_v_list[t][i][j])+",")
							#test_num += 1
						a_v_list_test_f.write("\n")
		a_v_list_train_f.close()
		a_v_list_test_f.close()

		# print("a_v_train_num: " + str(train_num))
		# print("a_v_test_num: " + str(test_num))
		# print (float(test_num) / test_a_num)



input_data_class = input_data(args = args)


input_data_class.gen_het_rand_walk()


#input_data_class.gen_meta_rand_walk_APVPA()


#input_data_class.a_a_collaborate_train_test() #set author-author collaboration data 


#input_data_class.a_p_citation_train_test() #set author-paper citation data 


#input_data_class.a_v_train_test() #generate author-venue data 



