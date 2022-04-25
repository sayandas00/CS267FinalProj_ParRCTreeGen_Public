import sys
import getopt


def generate_chain(num, filename):
    print("N = " + str(num) + ", M = " + str(num-1))
    f = open(filename, 'w')
    f.write(str(num) + " " + str(num-1) + "\n")
    for i in range(1, num):
        f.write(str(i) + " " + str(i+1) + " 1\n")
        
def generate_binary(num, filename):
    print("N = " + str(num) + ", M = " + str(num-1))
    f = open(filename, 'w')
    f.write(str(num) + " " + str(num-1) + "\n")
    for i in range(1, int(num/2)+1):
        f.write(str(i) + " " + str(i*2) + " 1\n")
        if i*2+1 <= num:
            f.write(str(i) + " " + str(i*2+1) + " 1\n")
        
def generate_file_connected(num, infile, outfile):
    inf = open(infile, 'r')
    init = inf.readline().split(" ")
    init_v = int(init[0])
    init_e = int(init[1])

    num_iters = int(num/init_v)+1
    total_v = num_iters*init_v

    total_e = num_iters*init_e + (num_iters - 1)

    init_graph = []
    for i in range(init_e):
        init_graph.append(inf.readline().split(" "))
    
    ouf = open(outfile, 'w')
    ouf.write(str(total_v) + " " + str(total_e) + "\n")
    for i in range(0, num_iters):
        for j in range(init_e):
            ouf.write(str( (i*init_v)+int(init_graph[j][0]) ) + " " + str( (i*init_v)+int(init_graph[j][1]) ) + " 1\n" )
        if i != num_iters-1:
            ouf.write( str(init_v*(i+1)) + " " + str( init_v*(i+1) + 1 ) + " 1\n" )
    
        
def main():
    inputfile=''
    outfile=''
    type=''
    num=-1
    try:
        opts, args = getopt.getopt(sys.argv[1:],"ht:i:o:n:",["type=","ifile=","ofile=","num="])
    except getopt.GetoptError:
        print("test.py -t <type> -o <outputfile> -n <num vertices>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("test.py -t <type> -o <outputfile> -n <num vertices>")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-t", "--type"):
            type = arg
        elif opt in ("-n", "--num"):
            num=int(arg)
            
    if type=="chain":
        generate_chain(num, outputfile)
    elif type=="binomial":
        generate_binary(num, outputfile)
    elif type=="file":
        generate_file_connected(num, inputfile, outputfile)
    else:
        print("type not recognized")
        sys.exit(2)
        
if __name__ == "__main__":
    main()
        
    