import re, math
from collections import Counter

WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
     # 检查这两个向量之间的一样的字符
     intersection = set(vec1.keys()) & set(vec2.keys()) #{'call'}
     # 分子：对序列进行求和,如果intersection为空则为0，相反则为1
     numerator = sum([vec1[x] * vec2[x] for x in intersection])   # 1 / 0

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     # 分母
     denominator = math.sqrt(sum1) * math.sqrt(sum2) # 2.0
     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     # ['This', 'program', 'produces', 'call']
     # ['call']

     return Counter(words)
     # {'This': 1, 'program': 1, 'produces': 1, 'call': 1}
     # {'call': 1}





