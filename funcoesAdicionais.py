from math import floor
from math import ceil
from random import sample
import shutil
import os
import os.path
from Imagem import *
from Pastas import *

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def criaDir (caminho):
    retorno = False
    try:
        os.makedirs(caminho, mode=0o777, exist_ok=False)
    except OSError:
        print('Creation of the directory "%s" failed' % caminho)
    else:
        print('Successfully created the directory "%s"' % caminho)
        retorno = True
    return retorno

def handler(func, path, exc_info):
    print("Inside handler")
    print(exc_info)

def remocao (caminho):
    try:
        #   shutil.rmtree(caminho, ignore_errors=True)
        shutil.rmtree(caminho, onerror=handler)
    except OSError as e:
        print("Error: %s : %s" % (caminho, e.strerror))

def verificaDir(diretorio):
    var = criaDir(diretorio)
    if (not var):
        remocao(diretorio)
        var = criaDir(diretorio)
    return var

def novoDir (caminho, diretorio):
    aT = []
    print("\nRenomeando...")

    var = verificaDir(diretorio)

    if (var):
        for pasta in os.listdir(caminho):
            subpasta = os.path.join(caminho, pasta)
            cSP = format(os.path.basename(subpasta))
            dest_dir = diretorio+"/"+cSP  #   cria a sub pasta no destino
            criaDir(dest_dir)
            #   get the current working dir
            #   src_dir = os.getcwd()
            for image_file in os.listdir(subpasta):
                fsp = image_file.split(".")
                ext = fsp[len(fsp) - 1]
                #   ext = "."+fsp[len(fsp)-1]
                if ext in ALLOWED_EXTENSIONS:
                    src_file = os.path.join(subpasta, image_file)
                    shutil.copy(src_file, dest_dir)  # copy the file to destination dir
                    nomeF = cSP+"_"+image_file  #   nome final

                    dst_file = os.path.join(dest_dir, image_file)
                    new_dst_file_name = os.path.join(dest_dir, nomeF)

                    os.rename(dst_file, new_dst_file_name)  # rename
                    aT.append(Imagem(nomeF, new_dst_file_name, cSP, nomeF, new_dst_file_name, cSP, False))
                    #   path = os.path.dirname(src_file)
                    #   print(src_file)
                    #   print("Arquivo: {}".format(image_file))
    else:
        aT = lerAP(diretorio)
    return aT

def criaDirTreinamento (caminho, qtd, treino, diretorio, nomeT, aT, output, txtNome):
    print("\n\nCriando diretório de testes\n")
    print("Movendo as imagens...")
    i = 1

    tam = len(str(len(aT)))

    var = verificaDir(diretorio)
    if var:
        verificaDir(output)
        fh = open(output + '/' + txtNome, 'w')
        for pasta in os.listdir(caminho):
            subpasta = os.path.join(caminho, pasta)
            arrayImg = os.listdir(subpasta)
            file_count = len(arrayImg)
            qtdImg = floor((file_count*qtd)/100)
            if treino:
                qtdImg = ceil((file_count*qtd)/100)
            sortedArrayImg = sample(arrayImg, qtdImg)

            for img in sortedArrayImg:
                src_file = os.path.join(subpasta, img)
                shutil.move(src_file, diretorio)  # copy the file to destination dir

                dst_file = os.path.join(diretorio, img)
                nomeArq = img.split(".")
                formato = nomeArq[len(nomeArq)-1]
                novoNome = nomeT+"_"+str(i).zfill(tam)+"."+formato
                new_dst_file_name = os.path.join(diretorio, novoNome)

                os.rename(dst_file, new_dst_file_name)  # rename

                findObject(img, aT, novoNome, new_dst_file_name, diretorio)

                fh.write(img+"\t"+novoNome+"\t"+new_dst_file_name+"\t"+diretorio+"\n")

                i += 1
        fh.close()
    else:
        f = open(output+'/'+txtNome, 'r')
        linha = f.readlines()
        for x in linha:
            arr = x.split("\t")
            var = findObject(arr[0], aT, arr[1], arr[2], arr[3])
            if not var:
                arq = arr[0].split("_")
                aT.append(Imagem(arr[0], caminho+"/"+arq[0]+"/"+arr[0], arq[0], arr[1], arr[2], arr[3], True))
                aT = sorted(aT, key=lambda h: h.nomeO)
        f.close()
    pastaInfo(aT)

def lerAP (diretorio):    #   lê arquivos da pasta e retorna um array com os endereços dos arquivos
    aI = []
    for pasta in os.listdir(diretorio):
        full_path = os.path.join(diretorio, pasta)
        for img in os.listdir(full_path):
            full_file_path = os.path.join(full_path, img)
            folder = format(os.path.basename(full_path))
            aI.append(Imagem(img, full_file_path, folder, img, full_file_path, folder, False))
    return aI

def permuta (aT, pastasL, inter, interA, treino, aQ, notOk):
    for x in pastasL:
        if interA == 0:
            aQ, notOk = divInt(x.qtdT, inter)
        if not(notOk):
            imagensT = []
            imagensTr = []
            for y in aT:
                if (y.pastaO == x.nome):
                    if (y.treino == treino):
                        imagensTr.append(y)
                    else:
                        imagensT.append(y)
            sortedImagensT = sample(imagensT, int(aQ[interA]))
            sortedImagensTr = sample(imagensTr, int(aQ[interA]))
            for z in range(int(aQ[interA])):
                diretorioT = sortedImagensT[z].diretorioM
                diretorioTr = sortedImagensTr[z].diretorioM
                nomePerm = sortedImagensTr[z].nomeM

                fsp = diretorioT.split("\\")

                shutil.move(diretorioTr, fsp[0])

                os.rename(fsp[0]+"/"+nomePerm, fsp[0]+"/"+sortedImagensTr[z].nomeO)


                diretorioPerm = sortedImagensTr[z].pastaM

                sortedImagensTr[z].diretorioM = sortedImagensTr[z].diretorioO
                sortedImagensTr[z].nomeM = sortedImagensTr[z].nomeO
                sortedImagensTr[z].pastaM = sortedImagensTr[z].pastaO
                sortedImagensTr[z].treino = not(sortedImagensTr[z].treino)

                print("1\t-\tFoi movido:\t"+diretorioTr+"\tpara\t"+fsp[0])
                print("1\t-\tO arquivo:\t"+fsp[0]+"/"+nomePerm+"\tfoi renomeado para\t"+sortedImagensTr[z].nomeO+"\t"+str(sortedImagensTr[z].treino))

                shutil.move(diretorioT, diretorioPerm)

                os.rename(diretorioPerm+"/"+sortedImagensT[z].nomeO, diretorioPerm+"/"+nomePerm)

                sortedImagensT[z].diretorioM = diretorioPerm + "/" + nomePerm
                sortedImagensT[z].nomeM = nomePerm
                sortedImagensT[z].pastaM = diretorioPerm
                #   sortedImagensT[z].treino = not (sortedImagensTr[z].treino)

                print("2\t-\tFoi movido:\t" + diretorioT + "\tpara\t" + diretorioPerm)
                print("2\t-\tO arquivo:\t" + diretorioPerm + "/" + sortedImagensT[z].nomeO + "\tfoi renomeado para\t" +nomePerm+"\t"+str(not(sortedImagensT[z].treino)))
        else:
            print("A permuta falhou!\nAs interações não são suficientes para a execução")
            break
    return aQ

def pastaInfo (aT):
    lista = []
    past = []
    pastas = set()
    folders = []
    for x in aT:
        folders.append(x.pastaO)
        if not(x.treino):
            past.append(x.pastaO)
            pastas.add(x.pastaO)
    for y in pastas:
        lista.append(Pastas(y, folders.count(y), past.count(y)))
    return lista

def findObject (nomeO, aT, nomeM, diretorioM, pastaM):
    for x in aT:
        if x.nomeO == nomeO:
            x.nomeM = nomeM
            x.diretorioM = diretorioM
            x.pastaM = pastaM
            x.treino = True
            return True
    return False

def imprimeImg (aT):
    for obj in aT:
            print(obj.nomeO, obj.diretorioO, obj.pastaO, obj.nomeM, obj.diretorioM, obj.pastaM, obj.treino, sep='\t-\t')

def imprimePasta (pastaData):
    for obj in pastaData:
        print(obj.nome, obj.qtd, obj.qtdT, obj.qtdTr, sep='\t-\t')

def divInt (num, div):
    arr = []
    notOk = False
    if(num >= div and div != 0):
        if(num % div != 0):
            for x in range(div):
                if x == div-1:
                    arr.append(floor(num / div)+1)
                else:
                    arr.append(floor(num/div))
        else:
            for x in range(div):
                arr.append(num / div)
    else:
        notOk = True
        print("Não foi possível atender a solicitação")
    return arr, notOk