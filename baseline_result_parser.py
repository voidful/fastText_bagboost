
import pprint
f = open('baseline.detail','r')   
f.readline()


counter = 1
dic = {}
l = []
for line in f:    
    if counter%7 ==1: 
    	dic["question"] = str(line).rstrip()  
    elif counter%7 ==2:
    	dic["predict label"] = str(line).split()[2]
    elif counter%7 ==3:
    	dic["predict prob"] = float(line.rstrip().split()[2])
    elif counter%7 ==4:
    	dic["true label"] = str(line).split()[2]
    elif counter%7 ==5:
    	dic["correct"] = str(line).split()[2]
    elif counter%7 ==6:
    	dic["entropy"] = float(line.rstrip().split()[2])
    elif counter%7 ==0:
    	l.append(dic)
    	dic = {}
    counter=counter+1
f.close() 


pp = pprint.PrettyPrinter(indent=4)

def predicting_statstics(threshold, l):
    l = list(filter(lambda d: d["entropy"]>threshold, l))
    l.sort(key=lambda d:d['entropy'], reverse=True)
    print("entropy > %s 的資料有 %s 筆"%(threshold,len(l)))

predicting_statstics(3,l)
predicting_statstics(2.5,l)
predicting_statstics(2,l)
predicting_statstics(1.5,l)
predicting_statstics(1,l)
predicting_statstics(0.5,l)
predicting_statstics(0,l)

l.sort(key=lambda d:d['entropy'], reverse=True)
l = list(filter(lambda d: d["entropy"]>1.5, l))
# pp.pprint(l)

from itertools import groupby
l.sort(key=lambda d:d['true label'], reverse=True)
histogram = []
for k, g in groupby(l, lambda x: x["true label"]):
   histogram.append([k, len(list(g))])

histogram.sort(key = lambda x:x[1])
pp.pprint(histogram)
