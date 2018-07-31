import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer

testdata = pd.DataFrame({
    'age': [4 , 6, 3, 3],
    'salary': [4, 5, 1, 1],
    'pet': ['cat', 'dog', 'dog', 'fish']
})

# 对数值型输入
a1 = OneHotEncoder(sparse = False).fit_transform( testdata[['age']] )
a2 = OneHotEncoder(sparse = False).fit_transform( testdata[['salary']])

a3 = OneHotEncoder(sparse = False).fit_transform( testdata[['age','salary']] )
print(a3)

# 对字符串型输入
a4 = LabelBinarizer().fit_transform(testdata['pet'])
print(a4)

# 矩阵拼接
final_output = np.hstack((a3,a4))
print(final_output)
