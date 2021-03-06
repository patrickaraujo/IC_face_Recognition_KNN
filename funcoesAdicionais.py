from math import floor
from math import ceil
from random import sample
from PIL import Image
import shutil
import os
import os.path
from Imagem import *
from Pastas import *
import sys

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

def handler(exc_info):
    print("Inside handler")
    print(exc_info)

def remocao (caminho):
    try:
        #   shutil.rmtree(caminho, ignore_errors=True)
        shutil.rmtree(caminho, onerror=handler)
    except OSError as e:
        print("Error: %s : %s" % (caminho, e.strerror))

def verificaDir(diretorio, erase):
    var = criaDir(diretorio)
    if (not var and erase):
         remocao(diretorio)
         var = criaDir(diretorio)
    return var

def novoDirRenomeia (caminho, diretorio, erase):
    print("\nRenomeando...")

    var = verificaDir(diretorio, erase)

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
                    os.chmod(dst_file, 0o777)

                    new_dst_file_name = os.path.join(dest_dir, nomeF)

                    os.rename(dst_file, new_dst_file_name)  # rename
                    #   path = os.path.dirname(src_file)
                    #   print(src_file)
                    #   print("Arquivo: {}".format(image_file))

def criaDirTestes (caminho, diretorio, erase, contrl, div, reset):
    aT = []

    var = verificaDir(diretorio, erase)

    if (var):
        for pasta in os.listdir(caminho):
            subpasta = os.path.join(caminho, pasta)
            cSP = format(os.path.basename(subpasta))
            dest_dir = diretorio + "/" + cSP  # cria a sub pasta no destino
            criaDir(dest_dir)
            for image_file in os.listdir(subpasta):
                src_file = os.path.join(subpasta, image_file)
                shutil.copy(src_file, dest_dir)  # copy the file to destination dir
                dst_file = os.path.join(dest_dir, image_file)
                os.chmod(dst_file, 0o777)
                if contrl:
                    width, height = adImg(dst_file, 0, 0, div, reset)
                    adImg(dst_file, width, height, 0, reset)
                aT.append(Imagem(image_file, dst_file, cSP, image_file, dst_file, cSP, False, False))
    else:
        aT = lerAP(diretorio)
    return aT

def criaDirTreinamento (caminho, qtd, diretorio, nomeT, aT, output, erase, contrl, div, reset):
    array = aT

    print("\n\nCriando diretório de testes\n")
    print("Movendo as imagens...")
    i = 1

    tam = len(str(len(aT)))

    var = verificaDir(diretorio, erase)

    if var:
        for pasta in os.listdir(caminho):
            subpasta = os.path.join(caminho, pasta)
            arrayImg = os.listdir(subpasta)
            file_count = len(arrayImg)
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

                findObject(img, array, novoNome, new_dst_file_name, diretorio)

                i += 1
            if contrl:
                for img in arrayImg:
                    if img not in sortedArrayImg:
                        src_file = os.path.join(subpasta, img)
                        width, height = adImg(src_file, 0, 0,div, reset)
                        adImg(src_file, width, height, 0, reset)


    else:
        arrayFiles = sorted(os.listdir(output), reverse=True)

        f = open(output + '/' + arrayFiles[0], 'r')
        linha = f.readlines()
        for img in os.listdir(diretorio):
            for x in linha:
                arr = x.split("\t")
                if arr[3] == img:
                    ult = arr[7].split("\n")
                    array.append(Imagem(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], True, True))
                    break
        f.close()
        verificaDir(output, True)


        array = sorted(array, key=lambda h: h.nomeO)
    return array

def lerAP (diretorio):    #   lê arquivos da pasta e retorna um array com os endereços dos arquivos
    aI = []
    for pasta in os.listdir(diretorio):
        full_path = os.path.join(diretorio, pasta)
        for img in os.listdir(full_path):
            full_file_path = os.path.join(full_path, img)
            aI.append(Imagem(img, full_file_path, pasta, img, full_file_path, pasta, False, False))
    return aI

def permuta (aT, pastasL, inter, interA, done, aQ, contrl, div, reset, amostras):
    for x in pastasL:
        aQ, notOk = divInt(x.qtdT, inter)
        if not(notOk):
            imagensT = []
            imagensTr = []
            for y in aT:
                if (y.pastaO == x.nome):
                    if (y.done == done and y.treino == done):
                        imagensTr.append(y)
                    if (y.done != done and y.treino != done):
                        imagensT.append(y)
            sortedImagensT = sample(imagensT, int(aQ[interA]))
            sortedImagensTr = sample(imagensTr, int(aQ[interA]))
            for z in range(int(aQ[interA])):
                diretorioT = sortedImagensT[z].diretorioM.replace('\\', '/')

                nomePerm = sortedImagensTr[z].nomeM
                fsp = diretorioT.split("/")

                diretorioTr = sortedImagensTr[z].diretorioM
                pastD = fsp[0]+"/"+fsp[1]  #   pasta destino
                shutil.move(diretorioTr, pastD)

                pNDM = pastD+"/"+nomePerm   # primeiro novo diretório com nome modificado
                pNDO = pastD + "/" + sortedImagensTr[z].nomeO  # primeiro novo diretório com nome original

                os.rename(pNDM, pNDO)

                if contrl:
                    width, height = adImg(pNDO, 0, 0, div, reset)
                    adImg(pNDO, width, height, 0, reset)

                diretorioPerm = sortedImagensTr[z].pastaM

                sortedImagensTr[z].diretorioM = sortedImagensTr[z].diretorioO
                sortedImagensTr[z].nomeM = sortedImagensTr[z].nomeO
                sortedImagensTr[z].pastaM = sortedImagensTr[z].pastaO
                sortedImagensTr[z].treino = not(sortedImagensTr[z].treino)
                #   sortedImagensTr[z].done = not (sortedImagensTr[z].done)

                # print("1\t-\tFoi movido:\t"+diretorioTr+"\tpara\t"+fsp[0])
                # print("1\t-\tO arquivo:\t"+fsp[0]+"/"+nomePerm+"\tfoi renomeado para\t"+sortedImagensTr[z].nomeO+"\t"+str(sortedImagensTr[z].treino)+"\t"+str(sortedImagensTr[z].done))

                if contrl:
                    if os.path.exists(diretorioT):
                        os.remove(diretorioT)
                    else:
                        print("The file does not exist")
                    diretorioT = amostras+"/"+sortedImagensT[z].pastaO+"/"+sortedImagensT[z].nomeO
                    shutil.copy(diretorioT, diretorioPerm)
                else:
                    shutil.move(diretorioT, diretorioPerm)

                sNDM = diretorioPerm+"/"+sortedImagensT[z].nomeO    #   segundo novo diretório com nome modificado
                sNDO = diretorioPerm+"/"+nomePerm                   #   segundo novo diretório com nome original

                os.rename(sNDM, sNDO)

                sortedImagensT[z].diretorioM = sNDO
                sortedImagensT[z].nomeM = nomePerm
                sortedImagensT[z].pastaM = diretorioPerm
                sortedImagensT[z].treino = not (sortedImagensT[z].treino)
                sortedImagensT[z].done = True

                # print("2\t-\tFoi movido:\t" + diretorioT + "\tpara\t" + diretorioPerm)
                # print("2\t-\tO arquivo:\t" + diretorioPerm + "/" + sortedImagensT[z].nomeO + "\tfoi renomeado para\t" +nomePerm+"\t"+str(sortedImagensT[z].treino)+"\t"+str(sortedImagensT[z].done))
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
    lista = sorted(lista, key=lambda h: h.nome)
    return lista

def findObject (nomeO, aT, nomeM, diretorioM, pastaM):
    for x in aT:
        if x.nomeO == nomeO:
            x.nomeM = nomeM
            x.diretorioM = diretorioM
            x.pastaM = pastaM
            x.treino = True
            x.done = True
            return True
    return False

def exportListImg (aT, separador, nomeArq):

    fh = open(nomeArq, 'w')
    trigger = False
    sep = ""
    for obj in aT:
        if trigger:
            sep = "\n"
        fh.write(sep+obj.nomeO+separador+obj.diretorioO+separador+obj.pastaO+separador+obj.nomeM+separador+obj.diretorioM+separador+obj.pastaM+separador+str(obj.treino)+separador+str(obj.done))
        trigger = True
    fh.close()

def imprimeListImg (aT, separador):
    for obj in aT:
        print(obj.nomeO, obj.diretorioO, obj.pastaO, obj.nomeM, obj.diretorioM, obj.pastaM, obj.treino, obj.done, sep=separador)

def imprimeImg (obj, separador):
    if obj:
        print(obj.nomeO, obj.diretorioO, obj.pastaO, obj.nomeM, obj.diretorioM, obj.pastaM, obj.treino, obj.done, sep=separador)

def imprimeListPasta (pastaData, separador):
    for obj in pastaData:
        print(obj.nome, obj.qtd, obj.qtdT, obj.qtdTr, sep=separador)

def imprimePasta (obj, separador):
    if obj:
        print(obj.nome, obj.qtd, obj.qtdT, obj.qtdTr, sep=separador)

def divInt (num, div):
    arr = []
    notOk = False
    if(num >= div and div != 0):
        y = num / div
        w = num % div
        if(w != 0):
            for x in range(div):
                if x >= (div - w):
                    arr.append(floor(y)+1)
                else:
                    arr.append(floor(y))
        else:
            for x in range(div):
                arr.append(y)
    else:
        notOk = True
        print("Não foi possível atender a solicitação")
    return arr, notOk

def checkResults (aT, nomeAtual, nomeId):
    retorno = "\t-\t"
    for x in aT:
        if nomeAtual == x.nomeM:
            if x.pastaO == nomeId:
                return (retorno+str(True)+"\t-\t"+x.pastaO+"\t-\tFace Encontrada {}\t-\tAcertou".format(nomeId))
            return (retorno+str(False)+"\t-\t"+x.pastaO+"\t-\tFace Encontrada {}\t-\tErrou".format(nomeId))
    retorno += str(None)+"\t-\t"+str(None)+"\t-\tFace Nao Encontrada\t-\tErrou"
    return retorno

def adImg (arq, width, height, div, reset): #   Aumenta ou diminui
    img = Image.open(arq)
    widthO, heightO = img.size
    if(div > 0):
        width = int(widthO / div)
        height = int(heightO / div)
        resizeImg(img, arq, width, height)
    else:
        if reset:
            resizeImg(img, arq, width, height)
    return widthO, heightO

def resizeImg (img, arq , width, height):
    new_img = img.resize((width, height))
    new_img.save(arq, "JPEG", optimize=True)