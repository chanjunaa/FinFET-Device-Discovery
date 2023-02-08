import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import os
import xlwt
from tqdm import tqdm
from pathlib import Path



inputfile = r"D:\study\data\chenan\2022\Tellurene\ML\data\tell_bandgap_dataset.csv"
def main():
    df = pd.read_csv(inputfile,encoding='gbk')
    features = np.array(df.drop(['system'],1))
    model = joblib.load(r"D:\study\data\chenan\2022\Tellurene\ML\5fold-5.pkl")
    #print(model)
    results = model.predict(features)
    data = pd.DataFrame(results)

    writer = pd.ExcelWriter("D:\study\data\chenan\\2022\Tellurene\ML result\\5fold_5_result.xls")
    data.to_excel(writer, 'results', float_format='%.5f')

    writer.save()
    writer.close()

if __name__ == "__main__":
    main()

