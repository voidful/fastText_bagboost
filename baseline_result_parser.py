
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

l.sort(key=lambda d:d['entropy'], reverse=True)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(l)

