import arff, numpy

def abre_arff(caminho):
    """

    :param caminho: caminho arff
    :return: base
    """
    base=arff.load(open(caminho),encode_nominal=True)

    return base

def retorna_classes_existentes(dataset):
    '''

    :rtype: object
    :@param dataset: arff aberto
    :@return: n_classes, classes, elementos_p_classes, total_elementos
    '''

    #classes=[]#vetor com as classes


   # print(dataset['attributes'][-1][1])
    classes = numpy.arange(len(dataset['attributes'][-1][1]), dtype=int)
        #classes.append(int(i)-1)

   # print(classes)
   # exit(0)
    num_class=len(classes)


    return num_class, classes

def retorna_instacias(dataset,np_array=False):
    """

    :param dataset: arff carregado
    :return: X e y
    """
    y=[]
    classes=[]
    X=[]
    for i in dataset['data']:
        y.append(i[-1])
        X.append(i[:-1])
    if(np_array==True):
        X=numpy.array(X)
        y=numpy.array(y)
    #print(dataset['attributes'][-1][1])

    return X,y,dataset['data']

def cria_arff(info, data, classes,pasta, nome):

        """
        cria um arquivo arff no E:
        @:param info: descricao da base/ relacao
        @:param data: dados da base
        @:param classes: classes da base
        @:param nome: do arquivo a ser gerado
        :return:
        """
        f = [str(k) for k in classes]
        info['attributes'][-1] = ('Class', f)
        obj = {
            'description': info['description'],
            'relation': info['relation'],
            'attributes': info['attributes'],
            'data': data['data'],

        }
        #print(obj)
        arq1 = arff.dumps(obj)
        arq = open(pasta+nome+'.arff','w')
        arq.write(arq1)
        arq.close()


def retorna_dic_data(dataset):
    #print(dataset)
    out=dict()
    #out['data']=list()
    out['class']=list()
    out['data']=dataset['data']
    for i in range(len(dataset['attributes'])):

       # print(dataset['attributes'][i][0])
        out['class'].append(dataset['attributes'][i][0])
    return out
