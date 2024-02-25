import gc 

def test_emltreeslearner_fit_memory_leaks():
    import emltreeslearner
    import array
    import random
    import time
    X = array.array('f', [1,2,3,1,1,
                          1,2,1,1,1,
                          1,1,2,3,1,
                          1,2,3,1,2,
                          3,2,1,1,1])

    Y = array.array('f', [1,
                          0,
                          1,
                          0,
                          1])
    gc.collect()

    etl = emltreeslearner.EmlTreeLearner(1,4,1)
    etl.fit(X,Y,5)
    gc.collect()
    after_fit = gc.mem_alloc()
    etl.fit(X,Y,5)
    gc.collect()
    after_fit1 = gc.mem_alloc()
    etl.fit(X,Y,5)
    gc.collect()
    after_fit2 = gc.mem_alloc()
    assert after_fit2-after_fit1 == 0, after_fit2-after_fit1 #Next fit should not grow memory
    assert after_fit1-after_fit == 0, after_fit1-after_fit #Next fit should not grow memory
    del etl
    gc.collect()


def test_emltreeslearner_random_crush_test():
    import emltreeslearner
    import array
    import random
    import time
    Y = array.array('f', [1,
                          0,
                          1,
                          0,
                          1])
    for j in range(100):
        etl = emltreeslearner.EmlTreeLearner(1,4,1)
        Xl = []
        for x in range(25):
            Xl.append(random.randrange(1,100))
        X = array.array('f',Xl)
        etl.fit(X,Y,5)
        del etl
        gc.collect()

def test_save_load():
    import emltreeslearner
    import array
    import random
    import time
    X = array.array('f', [1,2,3,1,1,
                          1,2,1,1,1,
                          1,1,2,3,1,
                          1,2,3,1,2,
                          3,2,1,1,1])

    Y = array.array('f', [1,
                          0,
                          1,
                          0,
                          1])

    etl1 = emltreeslearner.EmlTreeLearner(1,4,1)
    print(etl1)
    etl1.fit(X,Y,5)
    print(etl1)
    with open('fit_model.csv', 'w') as f:
        emltreeslearner.save_model(etl1, f)

    etl2 = emltreeslearner.EmlTreeLearner(1,4,1)
    print(etl2)
    with open('fit_model.csv', 'r') as f:
        emltreeslearner.load_model(etl2, f)

    with open('fit_model2.csv', 'w') as f:
        emltreeslearner.save_model(etl2, f)

    f = open('fit_model.csv', 'r')
    serialized1=f.read()
    f.close()

    f = open('fit_model2.csv', 'r')
    serialized2=f.read()
    f.close()

    for i in range(len(serialized2)):
        assert serialized2[i]==serialized1[i], "Serialized models differ" #Next fit should not grow memory

    del etl1
    del etl2
    del X
    del Y
    gc.collect()

gc.enable()
test_emltreeslearner_fit_memory_leaks()
test_emltreeslearner_random_crush_test()
for i in range(100):
    m1 = gc.mem_free()
    test_save_load()
    print(m1-gc.mem_free())
