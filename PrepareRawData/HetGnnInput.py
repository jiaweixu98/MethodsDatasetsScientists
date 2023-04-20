# generate HetGNN input
# nohup python -u kg2rawdata.py > kg2rawdata.log 2>&1 &
import operator
from tqdm import tqdm
import pickle as pk

def generate_input_data():
    paper_author = pk.load( open(
        '../../data/HetGNNdata/paper_author.pkl', 'rb'))
    author_paper = pk.load( open(
        '../../data/HetGNNdata/author_paper.pkl', 'rb'))
    paper_bioentity = pk.load( open('../../data/HetGNNdata/paper_bioentity.pkl', 'rb'))
    bioentity_paper = pk.load( open('../../data/HetGNNdata/bioentity_paper.pkl', 'rb'))
    paper_dataset = pk.load( open('../../data/HetGNNdata/paper_dataset.pkl', 'rb'))
    dataset_paper = pk.load( open('../../data/HetGNNdata/dataset_paper.pkl', 'rb'))
    paper_method = pk.load( open(
        '../../data/HetGNNdata/paper_method.pkl', 'rb'))
    method_paper = pk.load( open('../../data/HetGNNdata/method_paper.pkl', 'rb'))
    paper_citing_cited = pk.load( open('../../data/HetGNNdata/pmid_citing_cited.pkl', 'rb'))
    paper_ebd = pk.load( open('../../data/HetGNNdata/paper_constrain_ebd.pkl', 'rb'))
    # 将id变为顺序数字index（Pcount）
    PID_trans_Pcount = {}
    Pcount_trans_PID = {}

    #paper从0开始
    count = 0
    for PID in paper_author.keys():
        PID_trans_Pcount[PID] = count
        Pcount_trans_PID[count] = PID
        count += 1
    pk.dump(PID_trans_Pcount, open('PID_trans_Pcount.pkl', 'wb'))
    pk.dump(Pcount_trans_PID, open('Pcount_trans_PID.pkl', 'wb'))
    # author 从0开始


    count = 0
    AID_trans_Acount = {}
    Acount_trans_AID = {}

    for AID in author_paper.keys():
        AID_trans_Acount[AID] = count
        Acount_trans_AID[count] = AID
        count += 1
    pk.dump(AID_trans_Acount, open('AID_trans_Acount.pkl', 'wb'))
    pk.dump(Acount_trans_AID, open('Acount_trans_AID.pkl', 'wb'))

    # bioentity 从0开始
    count = 0
    BID_trans_Bcount = {}
    Bcount_trans_BID = {}
    for BID in bioentity_paper.keys():
        BID_trans_Bcount[BID] = count
        Bcount_trans_BID[count] = BID
        count += 1
    pk.dump(BID_trans_Bcount, open('BID_trans_Bcount.pkl', 'wb'))
    pk.dump(Bcount_trans_BID, open('Bcount_trans_BID.pkl', 'wb'))


    # dataset 从0开始
    count = 0
    DID_trans_Dcount = {}
    Dcount_trans_DID = {}
    for DID in dataset_paper.keys():
        DID_trans_Dcount[DID] = count
        Dcount_trans_DID[count] = DID
        count += 1
    pk.dump(DID_trans_Dcount, open('DID_trans_Dcount.pkl', 'wb'))
    pk.dump(Dcount_trans_DID, open('Dcount_trans_DID.pkl', 'wb'))

    # method 从0开始
    count = 0
    MID_trans_Mcount = {}
    Mcount_trans_MID = {}
    for MID in method_paper.keys():
        MID_trans_Mcount[MID] = count
        Mcount_trans_MID[count] = MID
        count += 1
    pk.dump(MID_trans_Mcount, open('MID_trans_Mcount.pkl', 'wb'))
    pk.dump(Mcount_trans_MID, open('Mcount_trans_MID.pkl', 'wb'))

    paper_c2author_c = {}
    author_c2paper_c = {}
    paper_c2bio_c = {}
    bio_c2paper_c = {}
    paper_c2citation_c = {}
    paper_c2ebd = {}
    paper_c2dataset_c = {}
    dataset_c2paper_c = {}
    paper_c2method_c = {}
    method_c2paper_c = {}
    for k, v in paper_author.items():
        # k,PId; V AID；下面转化为数字符号；实际上只转换bioentity应该就可以，在此全部转换，以防万一
        paper_c2author_c[PID_trans_Pcount[k]] = list(
            map(lambda x: AID_trans_Acount[x], v))
    # 以下基本相同
    for k, v in author_paper.items():
        author_c2paper_c[AID_trans_Acount[k]] = list(
            map(lambda x: PID_trans_Pcount[x], v))

    for k, v in paper_bioentity.items():
        paper_c2bio_c[PID_trans_Pcount[k]] = list(
            map(lambda x: BID_trans_Bcount[x], v))

    for k, v in bioentity_paper.items():
        bio_c2paper_c[BID_trans_Bcount[k]] = list(
            map(lambda x: PID_trans_Pcount[x], v))
        
    for k, v in paper_dataset.items():
        paper_c2dataset_c[PID_trans_Pcount[k]] = list(
            map(lambda x: DID_trans_Dcount[x], v)) 


    for k, v in dataset_paper.items():
        dataset_c2paper_c[DID_trans_Dcount[k]] = list(
            map(lambda x: PID_trans_Pcount[x], v))

    for k, v in paper_method.items():
        paper_c2method_c[PID_trans_Pcount[k]] = list(
            map(lambda x: MID_trans_Mcount[x], v))

    for k, v in method_paper.items():
        method_c2paper_c[MID_trans_Mcount[k]] = list(
            map(lambda x: PID_trans_Pcount[x], v))


    for k, v in paper_citing_cited.items():
        paper_c2citation_c[PID_trans_Pcount[k]] = list(
            map(lambda x: PID_trans_Pcount[x], v))

    for k, v in paper_ebd.items():
        paper_c2ebd[PID_trans_Pcount[k]] = v

    # p_a_list_train_f = open("../../data/HetGNNdata/p_a_list_train.txt", "w")
    # sorted_paper_c2author_c = sorted(
    #     paper_c2author_c.items(), key=operator.itemgetter(0))
    # for k, v in dict(sorted_paper_c2author_c).items():
    #     p_a_list_train_f.write(str(k) + ":")
    #     for tt in range(len(v)-1):
    #         p_a_list_train_f.write(str(v[tt]) + ",")
    #     p_a_list_train_f.write(str(v[-1]))
    #     p_a_list_train_f.write("\n")
    # p_a_list_train_f.close()

    # a_p_list_train_f = open("../../data/HetGNNdata/a_p_list_train.txt", "w")
    # sorted_author_c2paper_c = sorted(
    #     author_c2paper_c.items(), key=operator.itemgetter(0))
    # for k, v in dict(sorted_author_c2paper_c).items():
    #     a_p_list_train_f.write(str(k) + ":")
    #     for tt in range(len(v)-1):
    #         a_p_list_train_f.write(str(v[tt]) + ",")
    #     a_p_list_train_f.write(str(v[-1]))
    #     a_p_list_train_f.write("\n")
    # a_p_list_train_f.close()

    # b_p_list_train_f = open("../../data/HetGNNdata/b_p_list_train.txt", "w")
    # sorted_bio_c2paper_c = sorted(
    #     bio_c2paper_c.items(), key=operator.itemgetter(0))
    # for k, v in dict(sorted_bio_c2paper_c).items():
    #     b_p_list_train_f.write(str(k) + ":")
    #     for tt in range(len(v)-1):
    #         b_p_list_train_f.write(str(v[tt]) + ",")
    #     b_p_list_train_f.write(str(v[-1]))
    #     b_p_list_train_f.write("\n")
    # b_p_list_train_f.close()

    # p_b_f = open("../../data/HetGNNdata/p_b.txt", "w")
    # sorted_paper_c2bio_c = sorted(
    #     paper_c2bio_c.items(), key=operator.itemgetter(0))
    # for k, v in dict(sorted_paper_c2bio_c).items():
    #     p_b_f.write(str(k) + ":")
    #     for tt in range(len(v)-1):
    #         p_b_f.write(str(v[tt]) + ",")
    #     p_b_f.write(str(v[-1]))
    #     p_b_f.write("\n")
    # p_b_f.close()


    # d_p_list_train_f = open("../../data/HetGNNdata/d_p_list_train.txt", "w")
    # sorted_dataset_c2paper_c = sorted(
    #     dataset_c2paper_c.items(), key=operator.itemgetter(0))
    # for k, v in dict(sorted_dataset_c2paper_c).items():
    #     d_p_list_train_f.write(str(k) + ":")
    #     for tt in range(len(v)-1):
    #         d_p_list_train_f.write(str(v[tt]) + ",")
    #     d_p_list_train_f.write(str(v[-1]))
    #     d_p_list_train_f.write("\n")
    # d_p_list_train_f.close()

    # p_d_f = open("../../data/HetGNNdata/p_d.txt", "w")
    # sorted_paper_c2dataset_c = sorted(
    #     paper_c2dataset_c.items(), key=operator.itemgetter(0))
    # for k, v in dict(sorted_paper_c2dataset_c).items():
    #     p_d_f.write(str(k) + ":")
    #     for tt in range(len(v)-1):
    #         p_d_f.write(str(v[tt]) + ",")
    #     p_d_f.write(str(v[-1]))
    #     p_d_f.write("\n")
    # p_d_f.close()


    # p_p_citation_list = open("../../data/HetGNNdata/p_p_citation_list.txt", "w")
    # sorted_paper_c2citation_c = sorted(
    #     paper_c2citation_c.items(), key=operator.itemgetter(0))
    # for k, v in dict(sorted_paper_c2citation_c).items():
    #     p_p_citation_list.write(str(k) + ":")
    #     for tt in range(len(v)-1):
    #         p_p_citation_list.write(str(v[tt]) + ",")
    #     p_p_citation_list.write(str(v[-1]))
    #     p_p_citation_list.write("\n")
    # p_p_citation_list.close()

    # p_abstract_embed = open("../../data/HetGNNdata/p_abstract_embed.txt", "w")
    # sorted_paper_c2ebd = sorted(paper_c2ebd.items(), key=operator.itemgetter(0))
    # p_abstract_embed.write(str(len(sorted_paper_c2ebd))+' 128'+"\n")
    # for k, v in dict(sorted_paper_c2ebd).items():
    #     p_abstract_embed.write(str(k)+' ')
    #     for tt in range(len(v)-1):
    #         p_abstract_embed.write(str(v[tt]) + " ")
    #     p_abstract_embed.write(str(v[-1]))
    #     p_abstract_embed.write("\n")
    # p_abstract_embed.close()
    
    m_p_list_train_f = open("../../data/HetGNNdata/m_p_list_train.txt", "w")
    sorted_method_c2paper_c = sorted(
        method_c2paper_c.items(), key=operator.itemgetter(0))
    for k, v in dict(sorted_method_c2paper_c).items():
        m_p_list_train_f.write(str(k) + ":")
        for tt in range(len(v)-1):
            m_p_list_train_f.write(str(v[tt]) + ",")
        m_p_list_train_f.write(str(v[-1]))
        m_p_list_train_f.write("\n")
    m_p_list_train_f.close()

    p_m_f = open("../../data/HetGNNdata/p_m.txt", "w")
    sorted_paper_c2method_c = sorted(
        paper_c2method_c.items(), key=operator.itemgetter(0))
    for k, v in dict(sorted_paper_c2method_c).items():
        p_m_f.write(str(k) + ":")
        for tt in range(len(v)-1):
            p_m_f.write(str(v[tt]) + ",")
        p_m_f.write(str(v[-1]))
        p_m_f.write("\n")
    p_m_f.close()

if __name__ == '__main__':
    generate_input_data()
