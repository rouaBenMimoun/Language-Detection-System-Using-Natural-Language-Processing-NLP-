from sklearn.feature_extraction.text import CountVectorizer
texts=["chien chat poisson ", "chien chat chat " , "poisson oiseau oiseau"]
cv = CountVectorizer()
cv_fit=cv.fit_transform(texts)
print(cv.get_feature_names_out())
print(cv_fit.toarray())


import re
text = "this is[] a! testing? text"
text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
text = re.sub(r'[[]]', ' ', text)
print(text)