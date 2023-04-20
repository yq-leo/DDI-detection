import pandas as pd

filename = 'Entity2Vec_sg_200_5_5_15_2_500_d5_randwalks.txt'
outname = 'Entity2Vec_sg_200_5_5_15_2_500_d5_randwalks.csv'

embedding_df = pd.read_csv(filename, sep = '\t')
embedding_df.to_csv(outname, index = 0, columns = embedding_df.columns)