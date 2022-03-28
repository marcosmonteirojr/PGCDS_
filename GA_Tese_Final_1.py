import collections

import Marff, numpy as np
import random, os
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from joblib import  Parallel, delayed
import Cpx, csv, time
def distance(first=False, population=None):
    global pop, name_individual, dist, bags, min_score, group, types, c, score, pred, pool
    dist = dict()
    dist['name'] = list()
    dist['dist'] = list()
    dist['diver'] = list()
    dist['score'] = list()
    dist['score_g'] = list()
    if (first == True and generation == 0):
        dist['name'] = pop
        r = Parallel(n_jobs=jobs)(delayed(parallel_distance2)(i, bags, group, types) for i in range(len(dist['name'])))
        c, score,  pred, pool = zip(*r)
    elif (first == False and population == None):
        start_ = name_individual - nr_individual
        for i in range(start_, name_individual):
            x = []
            x.append(i)
            dist['name'].append(x)
        r = Parallel(n_jobs=jobs)(
            delayed(parallel_distance2)(j, bags, group, types) for j in range(100, nr_individual + 100))
        c, score, pred, pool = zip(*r)
    elif (population != None):
        dist['name'] = population
        indices = []
        for i in population:
            indices.append(bags['name'].index(str(i[0])))
        r = Parallel(n_jobs=jobs)(delayed(parallel_distance2)(i, bags, group, types) for i in indices)
        c, score, pred, pool = zip(*r)
    dist['dist'] = Cpx.dispersion_line(c)
    dist['score'] = score
    d = diversity(pred, y_vali)
    dist['diver'] = Cpx.min_max_norm(d)
    dist['score_g']=Cpx.voting_classifier(pool,X_vali,y_vali)
    return

def diversity(pred, y):
    print(pred[0])
    print(y[0])
    pred=np.array(pred)
    d =Cpx.diversitys(y, pred)
    return d

def parallel_distance2(i, bags, group, types):
    global classifier, pred_, score_

    """
    :param i: lista de indices do bag a ser testado
    :param bags: lista com todos os bags
    :param group: lista com o name dos groups de complexidades ex: [overllaping,,,,,]
    :param types: lista com o name da complexidade
    :return: listas de complexidade, score do prorpio bag, score sobre a validacao, e a predicao sobre a validacao
    """
    indx_bag1 = bags['inst'][i]
    X_bag, y_bag = monta_arquivo(indx_bag1)
    cpx = (Cpx.complexity_data3(X_bag, y_bag, group, types))

    if classifier=="perceptron":
        classifier, score_, pred_ = Cpx.biuld_classifier(X_bag, y_bag, X_bag, y_bag,X_vali,y_vali)
    elif classifier=="tree":
        classifier, score_, pred_ = Cpx.biuld_classifier_tree(X_bag, y_bag, X_bag, y_bag, X_vali, y_vali)

    #######################################################################
    return cpx,  score_,  pred_, classifier

def parallel_score(i, bags):

    indx_bag1 = bags['inst'][i]
    X_bag, y_bag = monta_arquivo(indx_bag1)
    _, score_, _ = Cpx.biuld_classifier(X_bag, y_bag, X_vali, y_vali)
    return score_

def monta_arquivo(indx_bag):
    global X, y
    '''
    Recebe o indice de instancias de um bag
    :param indx_bag:
    :param vet_classes: false, retorna o vetor de classes
    :return: X_data, y_data
    '''
    X_data = []
    y_data = []
    for i in indx_bag:
        X_data.append(X[int(i)])
        y_data.append(y[int(i)])
    return X_data, y_data

def cross(ind1, ind2):
    '''
    Para funcionar os bags devem ter o mesmo tamanho
    :param ind1:
    :param ind2:
    :return:
    '''
    global name_individual, cont_crossover, nr_individual, bags, dispersion, ind_out1

    individual = False
    indx = bags['name'].index(str(ind1[0]))
    indx2 = bags['name'].index(str(ind2[0]))
    indx_bag1 = bags['inst'][indx]
    indx_bag2 = bags['inst'][indx2]
    _, y_data = monta_arquivo(indx_bag1)
    cont = 0

    while (individual != True):

        ind_out1 = short_cross(y_data, indx_bag1, indx_bag2)
        individual = verifica_data(ind_out1)
        cont = cont + 1
        if cont == 30:
            print("erro de numero de classes")
            exit(0)

    ind1[0] = name_individual
    ind2[0] = name_individual

    bags['name'].append(str(name_individual))
    bags['inst'].append(ind_out1)
    name_individual += 1

    if (dispersion == True):
        cont_crossover = cont_crossover + 1

        if (cont_crossover == nr_individual + 1):
            cont_crossover = 1
            distance(first=False, population=None)

    return creator.Individual(ind1), creator.Individual(ind2)

def cross2(ind1, ind2):
    global name_individual, cont_crossover, nr_individual, bags, dispersion

    individual = False
    indx = bags['name'].index(str(ind1[0]))
    indx2 = bags['name'].index(str(ind2[0]))
    indx_bag1 = bags['inst'][indx]
    indx_bag2 = bags['inst'][indx2]
    _, y_data = monta_arquivo(indx_bag1)

    cont = 0

    ###############################
    indx_bag1 = [int(i) for i in indx_bag1]
    indx_bag2 = [int(i) for i in indx_bag2]

    ###############################
    while (individual != True):
        ind_out1, _ = tools.cxMessyOnePoint(indx_bag1, indx_bag2)
        individual = verifica_data(ind_out1)
        if cont == 30:
            print("erro de numero de classes")
            exit(0)
    ind_out1 = [str(i) for i in ind_out1]
    #print(ind_out1)
    ind1[0] = name_individual
    ind2[0] = name_individual
    print(ind_out1)
    bags['name'].append(str(name_individual))
    bags['inst'].append(ind_out1)
    name_individual += 1

    if (dispersion == True):
        cont_crossover = cont_crossover + 1

        if (cont_crossover == nr_individual + 1):
            cont_crossover = 1
            distance(first=False, population=None)

    return creator.Individual(ind1), creator.Individual(ind2)

def short_cross(y_data, indx_bag1, indx_bag2):
    start_ = fim = 0
    ind_out1 = []
    while (y_data[start_] == y_data[fim]):
        start_ = random.randint(0, len(y_data) - 1)
        fim = random.randint(start_, len(y_data) - 1)
    for i in range(len(y_data)):
        if (i <= start_ or i >= fim):
            ind_out1.append(indx_bag1[i])
        else:
            ind_out1.append(indx_bag2[i])
    return ind_out1

def verifica_data(ind_out):
    global classes
    _, y = monta_arquivo(ind_out)
    counter = collections.Counter(y)
    if len(counter.values()) == len(classes) and min(counter.values()) >= 2:
        return True
    else:
        return False

def mutate(ind):
    global off, name_individual, cont_crossover, nr_individual, bags, dispersion
    print("mutate")
    ind_out = []
    indx = bags['name'].index(str(ind[0]))
    indx_bag1 = bags['inst'][indx]
    X, y_data = monta_arquivo(indx_bag1)
    inst = 0
    inst2 = len(y_data)

    if (generation == 0 and off == []):
        ind2 = random.randint(0, 99)

    else:

        ind2 = random.sample(off, 1)
        ind2 = ind2[0]

    indx2 = bags['name'].index(str(ind2))
    indx_bag2 = bags['inst'][indx2]
    X2, y2_data = monta_arquivo(indx_bag2)

    while y_data[inst] != y2_data[inst2 - 1]:
        inst = random.randint(0, len(y_data) - 1)

    for i in range(len(indx_bag1)):
        if (i == inst):
            ind_out.append(indx_bag2[i])
        else:
            ind_out.append(indx_bag1[i])

    bags['name'].append(str(name_individual))
    bags['inst'].append(ind_out)

    ind[0] = name_individual
    name_individual += 1
    if (dispersion == True):
        cont_crossover = cont_crossover + 1
        if (cont_crossover == nr_individual + 1):
            cont_crossover = 1
            distance(first=False, population=None)

    return ind,

def fitness_andre(ind1):
    global dist, min_score
    for i in range(len(dist['name'])):
        if (dist['name'][i][0] == ind1[0]):
            dst = dist['dist'][i]
            ###########################
            score = dist["score"][i]
            break
    out = dst + score
    return out,

def fitness_dispercion_diver(ind1):
    global dist, min_score
    for i in range(len(dist['name'])):
        if (dist['name'][i][0] == ind1[0]):
            dst = dist['dist'][i]
            ###########################
            disv= dist["diver"][i]
            #print(dist['name'][i][0], dst, score)
            break
    ###############################
    return dst, disv,

def fitness_dispercion(ind1):
    global dist, min_score
    for i in range(len(dist['name'])):
        if (dist['name'][i][0] == ind1[0]):
            dst = dist['dist'][i]
            ###########################
            score = dist["score"][i]
            break
            ###########################
    return dst, score

def fitness_dispercion_line(ind1):
    global dist
    for i in range(len(dist['name'])):

        if (dist['name'][i][0] == ind1[0]):
            dst1 = dist['dist'][i][0]
            dist2 = dist['dist'][i][1]
            ###########################
            score = dist["score"][i]
            diver=dist['diver'][i]
            break
    ###############################
    return dst1, dist2, diver, score,

def continus_():
    global seq
    seq += 1
    return seq

def the_function(population, gen, fitness):

    '''
    responsavel por alterar a generation, assim como zerar variaveis, alterar populacoes, e copiar arquivos
    :param population: populacao, retorna do DEAP
    :param gen: generation Retorna do DEAP
    :param offspring: nova populacao
    :return:
    '''
    print("tf")
    global generation, off, dispersion, nr_generation, bags, local, file_out, accuracia_ant, \
        s, c, dist_temp, gen_temp, pop_temp, bags_temp, parada, save_info, generations_escolhida, base_name, fitness_temp

    generation = gen
    ###############################################333

    ###################################################3
    print("the_fuction")

    off = []
    base_name = base_name + str(generation)
    bags_ant = bags
    bags = dict()
    bags['name'] = list()
    bags['inst'] = list()
    for j in population:
        indx = bags_ant['name'].index(str(j[0]))
        bags['name'].append(bags_ant['name'][indx])
        bags['inst'].append(bags_ant['inst'][indx])
    del bags_ant
    for i in range(len(population)):
        off.append(population[i][0])
    if stop=="maxdistance":
        max_distance(fitness, generation=generation, population=off, bags=bags)
    elif stop =="maxacc":
        max_acc(fitness, dist['score_g'], generation=generation, population=off, bags=bags)
    if generation == nr_generation:
        if stop=="maxdistance":
            save_bags(pop_temp, bags_temp, gen_temp, base_name, type=1, generations_escolhida=generations_escolhida)
        elif stop=="maxacc":
            save_bags(pop_temp, bags_temp, gen_temp, base_name, type=2, generations_escolhida=generations_escolhida)
        else:
            save_bags(off,bags,base_name=base_name,type=0)
    if (dispersion == True and generation != nr_generation):
        distance(population=population)
    return population



def save_bags(pop_temp, bags_temp, gen_temp=None, base_name=None, type=0,generations_escolhida="x"):
    '''
    :param pop_temp: populacao a ser gravada, geralmente o off
    :param bags_temp: bags a serem gravados, geralmente os bags da the function ou bags da max (bags_temp)
    :param gen_temp: generations atual, ou a generations escolhida (melhor generations) isso soma no name do arquivo final
    :param base_name: nesse caso o numero generations junto ao name da base (isso soma no arquivo de saida (name do arquivo final))
    :@param name_arq_generations: o name do arquivo que grava a geacao escolhida OBRIGATORIO nos types 1 e 2
    :param type: primeiro type (0) a populacao final tradicional, type (1) populacao da distance average, type (2) populacao da acuracia global
    :return:
    '''
    global file_out, iteration
    if type==0:
        for j in pop_temp:
            name = []
            indx = bags['name'].index(str(j))
            nm = bags['inst'][indx]
            name.append(bags['name'][indx])
            name.extend(nm)
            Cpx.save_bag(name, 'bags', local + "/Bags", base_name + file_out, iteration)
    elif(type==1):
        x = open(generations_escolhida, "a")
        x.write(base_name + ";" + str(gen_temp) + "\n")
        x.close()
        for j in pop_temp:
            name = []
            indx = bags_temp['name'].index(str(j))
            nm = bags_temp['inst'][indx]
            name.append(bags_temp['name'][indx])
            name.extend(nm)
            if classifier=="perc":
                Cpx.save_bag(name, 'bags', local + "/Bags", base_name + file_out, iteration)
            elif classifier=="tree":
                Cpx.save_bag(name, 'bags', local + "/tree/Bags", base_name + file_out, iteration)

    elif type==2:
        x = open(generations_escolhida, "a")
        x.write(base_name + ";" + str(gen_temp) + "\n")
        x.close()
        for j in pop_temp:
            name = []
            indx = bags_temp['name'].index(str(j))
            nm = bags_temp['inst'][indx]
            name.append(bags_temp['name'][indx])
            name.extend(nm)
            Cpx.save_bag(name, 'bags', local + "tree/Bags", base_name + file_out, iteration)

def max_distance(fitness, generation=None, population=None, bags=None):
    '''
    escolha da maior distance entre os bags
    :param fit1: fittnes 1
    :param fit2:
    :param fit3:
    ideal para distance line
    '''
    global dist_temp, pop_temp, gen_temp, bags_temp, fitness_temp
    fitness_temp[0] = fitness[0]
    fitness_temp[1] = fitness[1]
    if fitness[2]:
        fitness_temp[2] = fitness[2]
        dist_dist_average = np.mean(Cpx.dispersion(np.column_stack([fitness[0], fitness[1], fitness[2]])))
        #print(Cpx.dispersion2(np.column_stack([fitness[0], fitness[1], fitness[2]])))
    else:
        dist_dist_average = np.mean(Cpx.dispersion(np.column_stack([fitness[0], fitness[1]])))
    if dist_dist_average > dist_temp:
        dist_temp = dist_dist_average
        pop_temp = population
        gen_temp = generation
        bags_temp = bags

def max_acc(fitness, acc,generation=None, population=None, bags=None):
    '''
    Maior acuracia entre as generation
    :param fit1: fittnes 1
    :param fit2:
    :param fit3:
    :param population: popoluacao atual geralmente o off
    :param bags: bags atuais
    :return:
    ideal para distance line
    '''
    global pop_temp, gen_temp, bags_temp, acc_temp, fitness_temp
    print(fitness[0])
    fitness_temp[0] = fitness[0]
    fitness_temp[1] = fitness[1]
    if fitness[2]:
        fitness_temp[2] = fitness[2]
    if acc > acc_temp:
        acc_temp = acc
        pop_temp = population
        gen_temp = generation
        bags_temp = bags

tem2=[]
acc_temp=0
base_name = 'Wine'

local_dataset = "D:\\pesquisas\\Tese\\Bases4\\Dataset\\"
local = "D:\\pesquisas\\Tese\\Bases4\\"
cpx_dir = "D:\\pesquisas\\Tese\\Bases4\\bags\\"
#min_score = 0

#base_name=sys.arg[1]
#types=sys.argv[2]
#types=types.split(",")

########

group = ["overlapping", 'neighborhood', '', '', '', '']
types = ["F3", 'N3', '', '', '', '']


#todos as versoes finais utilizam a dispercion
dispersion = True

#quantidade de fitness e a proporcao
fit_value1 = 1.0
fit_value2 = 1.0
fit_value3 = -1.0

#numero de generation
nr_generation = 19
#numero de individuals por generations
nr_individual = 100
#tamanho da populacao
nr_pop=100
#probabilidade de cruzamento e mutacao
proba_crossover = 0.99
proba_mutation = 0.01
#numero de filhos por generations
nr_child=100
#contador de cruzamento
cont_crossover = 1
#numero de interacoes
iteration=21

dist_temp=0
#numeros de processadores em paralelo
jobs = 7
#type de método "maxacc maxdistance" nenhum
stop="maxacc"
#classifieres "tree" arvore de decisao ou perceptron
classifier="tree"#tree,perc
#name do arquivo de generation (qual generations foi escolhida)
generations_escolhida="generations_final"
#name do arquivo com os bags finais
file_out = "graficos_tese"+base_name

fitness_=fitness_dispercion_line
#variavel para salvar o fitnees
fitness_temp=[]
for t in range(1, iteration):

    off = []
    name_individual = 100
    iteration = t
    generation = 0
    seq = -1
    print("iteracao", t, base_name)
    ######
    #Abre os bags validate e teste
    #se nao exisitir cria
    arq_dataset = local_dataset + base_name + ".arff"
    arq_arff = Marff.abre_arff(arq_dataset)
    X, y, _ = Marff.retorna_instacias(arq_arff)
    _, classes = Marff.retorna_classes_existentes(arq_arff)
    if os.path.isfile(local + "Bags/" + str(iteration) + "/" + base_name + ".csv") == False:
        print("Criando Bags train e validacao da interacao: "+str(t))
        X_train, y_train, X_test, y_test, X_vali, y_vali = Cpx.routine_save_bags(local_dataset, local, base_name,
                                                                                     iteration)
        print("train, Teste, Validacao e Bags da primeira generations foram criados na pasta: "+local )
    else:
        _, validation = Cpx.open_test_vali(local , base_name, iteration)
        X_vali, y_vali = Cpx.biuld_x_y(validation, X, y)
        bags = Cpx.open_bag(cpx_dir + str(iteration) + "/", base_name)
    print("Criando AG ")
    #type de fitness e os fit values
    creator.create("FitnessMult", base.Fitness, weights=(fit_value1, fit_value2, fit_value3))#, fit_value2, fit_value3))#verificar
    creator.create("Individual", list, fitness=creator.FitnessMult)
    toolbox = base.Toolbox()
    toolbox.register("attr_item", continus_)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                      toolbox.attr_item, 1)
    population = toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    pop = toolbox.population(n=nr_pop)
    if dispersion == True:
         distance(first=True)
    #funcao de fitness: fitness_andre - DSOC, fitness_dispercion_diver _ diversity e distance, dispercion_line - dist dist diver
    toolbox.register("evaluate", fitness_)
    toolbox.register("mate", cross)
    toolbox.register("mutate", mutate)
    #funcao de selecao
    toolbox.register("select", tools.selNSGA2)
    ini = time.time()
    #loop GA
    pop = algorithms.eaMuPlusLambda(pop, toolbox, nr_child, nr_individual, proba_crossover, proba_mutation,
                                          nr_generation,
                                             generation_function=the_function)
#    os.system("rm -r /tmp/Rtmp*")