

def load_model(builder, f):
    for line in f:
        line = line.rstrip('\r')
        line = line.rstrip('\n')
        tok = line.split(',')
        kind = tok[0]
        if kind == 'r':
            root = int(tok[1])
            builder.addroot(root)
        elif kind == 'n':
            feature = int(tok[1])
            value = float(tok[2])
            left = int(tok[3])
            right = int(tok[4])
            builder.addnode(left, right, feature, value)
        else:        
            # unknown value
            pass
    builder.loadtreefrombuf()

def save_model(builder, f):
    builder.savetreetobuf()
    while(1):
        rootline=builder.readroot()
        if(rootline == None):
            break
        f.write("r,"+str(rootline[0])+"\n")

    while(1):
        nodeline=builder.readnode()
        if(nodeline == None):
            break
        f.write("n,"+str(nodeline[0])+","+str(nodeline[1])+","+str(nodeline[2])+","+str(nodeline[3])+"\n")
