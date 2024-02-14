# import emltreeslearner
# import array
# import gc

# X = [
# array.array('f', [1,2,3,1,1]),
# array.array('f', [1,2,1,1,1]),
# array.array('f', [1,1,2,3,1]),
# array.array('f', [1,2,3,1,2]),
# array.array('f', [3,2,1,1,1]),
# ]
# Y = [
# array.array('f', [1]),
# array.array('f', [0]),
# array.array('f', [1]),
# array.array('f', [0]),
# array.array('f', [1]),
# ]

# Y2 = array.array('f', [1,0,1,0,1])

# print(gc.mem_free())
# etl = emltreeslearner.EmlTreeLearner(1,4,1)

# etl.fit(X,Y2)
# etl.plot()

# f = open('data.txt', 'w')
# f.write(etl.serialize())
# f.close()

# etl2 = emltreeslearner.EmlTreeLearner(1,4,1)
# f = open('data.txt')
# emltreeslearner.load_model(etl2,f.read())
# f.close()
# gc.collect()
# print(gc.mem_free())
