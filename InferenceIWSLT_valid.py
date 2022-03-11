import sys
import torch
import os

modelname = sys.argv[1]
start = int(sys.argv[2])
end = int(sys.argv[3])

checkpointfolder = 'results/checkpoints/'
modelfolder = checkpointfolder + modelname + '/'
ensemblefolder = modelfolder + 'ensemblemodel/'

batch=100

if 'CMLM' in modelname:
    bleufolder = 'results/BLEU/' + modelname + '/bestvalid_iter10_LEN3/'
else:
    bleufolder = 'results/BLEU/' + modelname + '/bestvalid_beam4/'

os.system('mkdir -p {}'.format(bleufolder))
os.system('mkdir -p {}'.format(ensemblefolder))

try:
    validbleu = torch.load(bleufolder+'validbleu.pt')
    bestepoch = max(validbleu, key=lambda k: validbleu[k])
    bestbleu = validbleu[bestepoch]
    print('best validation bleu = {} at epoch {}'.format(bestbleu, bestepoch))
except:
    validbleu = {}
    bestepoch = start
    bestbleu = 0

jlist = [j for j in range(start, end + 1)]

for j in jlist:
    if j not in validbleu.keys():
        cpname = modelfolder + 'checkpoint{}.pt'.format(j)
        bleu = bleufolder + 'checkpoint{}_valid.out'.format(j)
        print('evaluating {}'.format(bleu))
        if 'IWSLTdeen' in modelname:
            if 'CMLM' in modelname:
                if 'distill' in modelname:
                    command = 'python generate.py data-bin/iwslt14_deen_jointdict_distill/ --gen-subset valid --task translation_lev --path {} --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --remove-bpe --iter-decode-force-max-iter --iter-decode-with-beam 3 --quiet | tee {}'.format(
                        cpname, batch, bleu)
                else:
                    command = 'python generate.py data-bin/iwslt14_deen_jointdict/ --gen-subset valid --task translation_lev --path {} --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --remove-bpe --iter-decode-force-max-iter --iter-decode-with-beam 3 --quiet | tee {}'.format(
                        cpname, batch, bleu)
            else:
                if 'distill' in modelname:
                    command = 'python generate.py data-bin/iwslt14_deen_jointdict_distill/ --gen-subset valid --path {} --beam 4 --batch-size 128 --remove-bpe --lenpen 0.3 --quiet | tee {}'.format(
                        cpname, bleu)
                else:
                    command = 'python generate.py data-bin/iwslt14_deen_jointdict/ --gen-subset valid --path {} --beam 4 --batch-size 128 --remove-bpe --lenpen 0.3 --quiet | tee {}'.format(
                        cpname, bleu)
        elif 'IWSLTende' in modelname:
            if 'CMLM' in modelname:
                if 'distill' in modelname:
                    command = 'python generate.py data-bin/iwslt14_ende_jointdict_distill/ --gen-subset valid --task translation_lev --path {} --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --remove-bpe --iter-decode-force-max-iter --iter-decode-with-beam 3 --quiet | tee {}'.format(
                        cpname, batch, bleu)
                else:
                    command = 'python generate.py data-bin/iwslt14_ende_jointdict/ --gen-subset valid --task translation_lev --path {} --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --remove-bpe --iter-decode-force-max-iter --iter-decode-with-beam 3 --quiet | tee {}'.format(
                        cpname, batch, bleu)
            else:
                if 'distill' in modelname:
                    command = 'python generate.py data-bin/iwslt14_ende_jointdict_distill/ --gen-subset valid --path {} --beam 4 --batch-size 128 --remove-bpe --lenpen 0.3 --quiet | tee {}'.format(
                        cpname, bleu)
                else:
                    command = 'python generate.py data-bin/iwslt14_ende_jointdict/ --gen-subset valid --path {} --beam 4 --batch-size 128 --remove-bpe --lenpen 0.3 --quiet | tee {}'.format(
                        cpname, bleu)
        os.system(command)
        with open(bleu, 'r') as f:
            lines = f.read().splitlines()
            lastline = lines[-1].replace(',', '').split()
            validbleu[j] = float(lastline[6])
            if bestbleu < float(lastline[6]):
                bestbleu = float(lastline[6])
                bestepoch = j
            print('best validation bleu {} at epoch {}'.format(bestbleu, bestepoch))

for n in [5, 10]:
    bestcplist = sorted(validbleu, key=lambda key: validbleu[key], reverse=True)[:n]
    print('top {} checkpoints:'.format(n))
    for i in bestcplist:
        print('epoch {}: validation bleu {}'.format(i, validbleu[i]))

    if 'CMLM' in modelname:
        bestensemble = modelfolder + 'LEN3_bestmodel_bestvalid_ensemble{}_epoch{}_{}.pt'.format(n, min(bestcplist), max(bestcplist))
    else:
        bestensemble = modelfolder + 'bestmodel_bestvalid_ensemble{}_epoch{}_{}.pt'.format(n, min(bestcplist), max(bestcplist))

    cpname = ensemblefolder + 'checkpoint{}.pt'.format(bestcplist[0])
    try:
        model = torch.load(cpname)
    except:
        os.system('cp {} {}'.format(modelfolder + 'checkpoint{}.pt'.format(bestcplist[0]), cpname))
        model = torch.load(cpname)

    for i in range(1, len(bestcplist)):
        cpname2 = ensemblefolder + 'checkpoint{}.pt'.format(bestcplist[i])
        try:
            model2 = torch.load(cpname2)
        except:
            os.system('cp {} {}'.format(modelfolder + 'checkpoint{}.pt'.format(bestcplist[i]), cpname2))
            model2 = torch.load(cpname2)

        for param in model['model']:
            if 'decoder.embed_tokens.weight' in param:
                pass
            else:
                model['model'][param].add_(model2['model'][param])
        del model2
    for param in model['model']:
        if 'decoder.embed_tokens.weight' in param:
            pass
        else:
            model['model'][param].div_(float(n))
    torch.save(model, bestensemble)

    del model
    if 'CMLM' in modelname:
        bleu = bleufolder + 'LEN3_bestmodel_bestvalid_ensemble{}_epoch{}_{}.out'.format(n, min(bestcplist), max(bestcplist))
    else:
        bleu = bleufolder + 'bestmodel_bestvalid_ensemble{}_epoch{}_{}.out'.format(n, min(bestcplist), max(bestcplist))
    print('evaluating {}'.format(bleu))

    if 'IWSLTdeen' in modelname:
        if 'CMLM' in modelname:
            if 'distill' in modelname:
                command = 'python generate.py data-bin/iwslt14_deen_jointdict_distill/ --gen-subset test --task translation_lev --path {} --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --iter-decode-force-max-iter --iter-decode-with-beam 3 --remove-bpe | tee {}'.format(
                    bestensemble, batch, bleu)
            else:
                command = 'python generate.py data-bin/iwslt14_deen_jointdict/ --gen-subset test --task translation_lev --path {} --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --iter-decode-force-max-iter --iter-decode-with-beam 3 --remove-bpe | tee {}'.format(
                    bestensemble, batch, bleu)
        else:
            if 'distill' in modelname:
                command = 'python generate.py data-bin/iwslt14_deen_jointdict_distill/ --path {} --beam 4 --batch-size 128 --remove-bpe --lenpen 0.3 | tee {}'.format(
                    bestensemble, bleu)
            else:
                command = 'python generate.py data-bin/iwslt14_deen_jointdict/ --path {} --beam 4 --batch-size 128 --remove-bpe --lenpen 0.3 | tee {}'.format(
                    bestensemble, bleu)
    if 'IWSLTende' in modelname:
        if 'CMLM' in modelname:
            if 'distill' in modelname:
                command = 'python generate.py data-bin/iwslt14_ende_jointdict_distill/ --gen-subset test --task translation_lev --path {} --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --iter-decode-force-max-iter --iter-decode-with-beam 3 --remove-bpe | tee {}'.format(
                    bestensemble, batch, bleu)
            else:
                command = 'python generate.py data-bin/iwslt14_ende_jointdict/ --gen-subset test --task translation_lev --path {} --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --iter-decode-force-max-iter --iter-decode-with-beam 3 --remove-bpe | tee {}'.format(
                    bestensemble, batch, bleu)
        else:
            if 'distill' in modelname:
                command = 'python generate.py data-bin/iwslt14_ende_jointdict_distill/ --path {} --beam 4 --batch-size 128 --remove-bpe --lenpen 0.3 | tee {}'.format(
                    bestensemble, bleu)
            else:
                command = 'python generate.py data-bin/iwslt14_ende_jointdict/ --path {} --beam 4 --batch-size 128 --remove-bpe --lenpen 0.3 | tee {}'.format(
                    bestensemble, bleu)

    os.system(command)
    command = './compound_split_bleu.sh {}'.format(bleu)
    os.system(command)

torch.save(validbleu, bleufolder+'validbleu.pt')
