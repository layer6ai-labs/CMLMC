import sys
import torch
import os

modelname = sys.argv[1]
start = int(sys.argv[2])
end = int(sys.argv[3])

checkpointfolder = 'results/checkpoints/'
ensemblemodelpath = checkpointfolder + 'ensemblemodel.pt'
batch=100

# print('decode not using BEAM')
for n in [5]:
    bestbleu = 0
    modelfolder = checkpointfolder + modelname + '/'
    if 'CMLM' in modelname:
        bleufolder = 'results/BLEU/' + modelname + '/ensemble{}_iter10_LEN3/'.format(n)
    else:
        bleufolder = 'results/BLEU/' + modelname + '/ensemble{}_beam4/'.format(n)
    os.system('mkdir -p {}'.format(bleufolder))

    bestepoch = start
    jlist = [j for j in range(start, end + 1)]

    for j in jlist:
        cpname = modelfolder + 'checkpoint{}.pt'.format(j)
        model = torch.load(cpname)
        for i in range(1, n):
            cpname2 = modelfolder + 'checkpoint{}.pt'.format(j - i)
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
        torch.save(model, ensemblemodelpath)
        del model
        bleu = bleufolder + 'checkpoint{}_ensemble{}.out'.format(j, n)
        print('evaluating {}'.format(bleu))

        if 'IWSLTdeen' in modelname:
            if 'CMLM' in modelname:
                if 'distill' in modelname:
                    command = 'python generate.py data-bin/iwslt14_deen_jointdict_distill/ --gen-subset test --task translation_lev --path {} --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --remove-bpe --iter-decode-force-max-iter --iter-decode-with-beam 3 --quiet | tee {}'.format(
                        ensemblemodelpath, batch, bleu)
                else:
                    command = 'python generate.py data-bin/iwslt14_deen_jointdict/ --gen-subset test --task translation_lev --path {} --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --remove-bpe --iter-decode-force-max-iter --iter-decode-with-beam 3 --quiet | tee {}'.format(
                        ensemblemodelpath, batch, bleu)
            else:
                if 'distill' in modelname:
                    command = 'python generate.py data-bin/iwslt14_deen_jointdict_distill/ --path {} --beam 4 --batch-size 128 --remove-bpe --lenpen 0.3 --quiet | tee {}'.format(
                        ensemblemodelpath, bleu)
                else:
                    command = 'python generate.py data-bin/iwslt14_deen_jointdict/ --path {} --beam 4 --batch-size 128 --remove-bpe --lenpen 0.3 --quiet | tee {}'.format(
                        ensemblemodelpath, bleu)
        elif 'IWSLTende' in modelname:
            if 'CMLM' in modelname:
                if 'distill' in modelname:
                    command = 'python generate.py data-bin/iwslt14_ende_jointdict_distill/ --gen-subset test --task translation_lev --path {} --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --remove-bpe --iter-decode-force-max-iter --iter-decode-with-beam 3 --quiet | tee {}'.format(
                        ensemblemodelpath, batch, bleu)
                else:
                    command = 'python generate.py data-bin/iwslt14_ende_jointdict/ --gen-subset test --task translation_lev --path {} --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --remove-bpe --iter-decode-force-max-iter --iter-decode-with-beam 3 --quiet | tee {}'.format(
                        ensemblemodelpath, batch, bleu)
            else:
                if 'distill' in modelname:
                    command = 'python generate.py data-bin/iwslt14_ende_jointdict_distill/ --path {} --beam 4 --batch-size 128 --remove-bpe --lenpen 0.3 --quiet | tee {}'.format(
                        ensemblemodelpath, bleu)
                else:
                    command = 'python generate.py data-bin/iwslt14_ende_jointdict/ --path {} --beam 4 --batch-size 128 --remove-bpe --lenpen 0.3 --quiet | tee {}'.format(
                        ensemblemodelpath, bleu)

        os.system(command)
        with open(bleu, 'r') as f:
            lines = f.read().splitlines()
            lastline = lines[-1].replace(',', '').split()
            if bestbleu < float(lastline[6]):
                bestbleu = float(lastline[6])
                bestepoch = j
            print('best bleu {} at epoch {}'.format(bestbleu, bestepoch))

    if 'CMLM' in modelname:
        bestensemble = modelfolder + 'LEN3_bestmodel_ensemble{}_epoch{}_{}.pt'.format(n, bestepoch-n+1, bestepoch)
    else:
        bestensemble = modelfolder + 'bestmodel_ensemble{}_epoch{}_{}.pt'.format(n, bestepoch - n + 1, bestepoch)
    cpname = modelfolder + 'checkpoint{}.pt'.format(bestepoch)
    model = torch.load(cpname)
    for i in range(1, n):
        cpname2 = modelfolder + 'checkpoint{}.pt'.format(bestepoch - i)
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
        bleu = bleufolder + 'LEN3_bestmodel_ensemble{}_epoch{}_{}.out'.format(n, bestepoch-n+1, bestepoch)
    else:
        bleu = bleufolder + 'bestmodel_ensemble{}_epoch{}_{}.out'.format(n, bestepoch - n + 1, bestepoch)
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
    #
    # if 'CMLM' in modelname:
    #     start=max(start, bestepoch-20)
    #
    #     bestbleu = 0
    #     bestepoch = start
    #     modelfolder = checkpointfolder + modelname + '/'
    #     bleufolder = 'results/BLEU/' + modelname + '/ensemble{}_iter10_BEAM4LEN1/'.format(n)
    #     os.system('mkdir -p {}'.format(bleufolder))
    #
    #     jlist = [j for j in range(start, end + 1)]
    #
    #     for j in jlist:
    #         cpname = modelfolder + 'checkpoint{}.pt'.format(j)
    #         model = torch.load(cpname)
    #         for i in range(1, n):
    #             cpname2 = modelfolder + 'checkpoint{}.pt'.format(j - i)
    #             model2 = torch.load(cpname2)
    #             for param in model['model']:
    #                 if 'decoder.embed_tokens.weight' in param:
    #                     pass
    #                 else:
    #                     model['model'][param].add_(model2['model'][param])
    #             del model2
    #         for param in model['model']:
    #             if 'decoder.embed_tokens.weight' in param:
    #                 pass
    #             else:
    #                 model['model'][param].div_(float(n))
    #         torch.save(model, ensemblemodelpath)
    #         del model
    #         bleu = bleufolder + 'checkpoint{}_ensemble{}.out'.format(j, n)
    #         print('evaluating {}'.format(bleu))
    #
    #         if 'IWSLTdeen' in modelname:
    #             if 'distill' in modelname:
    #                 command = 'python generate.py data-bin/iwslt14_deen_jointdict_distill/ --gen-subset test --task translation_lev --path {} --beam 4 --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --remove-bpe --iter-decode-force-max-iter --quiet | tee {}'.format(
    #                     ensemblemodelpath, batch, bleu)
    #             else:
    #                 command = 'python generate.py data-bin/iwslt14_deen_jointdict/ --gen-subset test --task translation_lev --path {} --beam 4 --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --remove-bpe --iter-decode-force-max-iter --quiet | tee {}'.format(
    #                     ensemblemodelpath, batch, bleu)
    #         elif 'IWSLTende' in modelname:
    #             if 'distill' in modelname:
    #                 command = 'python generate.py data-bin/iwslt14_ende_jointdict_distill/ --gen-subset test --task translation_lev --path {} --beam 4 --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --remove-bpe --iter-decode-force-max-iter --quiet | tee {}'.format(
    #                     ensemblemodelpath, batch, bleu)
    #             else:
    #                 command = 'python generate.py data-bin/iwslt14_ende_jointdict/ --gen-subset test --task translation_lev --path {} --beam 4 --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --remove-bpe --iter-decode-force-max-iter --quiet | tee {}'.format(
    #                     ensemblemodelpath, batch, bleu)
    #
    #         os.system(command)
    #         with open(bleu, 'r') as f:
    #             lines = f.read().splitlines()
    #             lastline = lines[-1].replace(',', '').split()
    #             if bestbleu < float(lastline[6]):
    #                 bestbleu = float(lastline[6])
    #                 bestepoch = j
    #             print('best bleu {} at epoch {}'.format(bestbleu, bestepoch))
    #
    #     bestensemble = modelfolder + 'BEAM4LEN1_bestmodel_ensemble{}_epoch{}_{}.pt'.format(n, bestepoch - n + 1, bestepoch)
    #     cpname = modelfolder + 'checkpoint{}.pt'.format(bestepoch)
    #     # model = torch.load(cpname, map_location=torch.device('cpu'))
    #     model = torch.load(cpname)
    #     for i in range(1, n):
    #         cpname2 = modelfolder + 'checkpoint{}.pt'.format(bestepoch - i)
    #         # model2 = torch.load(cpname2, map_location=torch.device('cpu'))
    #         model2 = torch.load(cpname2)
    #         for param in model['model']:
    #             if 'decoder.embed_tokens.weight' in param:
    #                 pass
    #             else:
    #                 model['model'][param].add_(model2['model'][param])
    #         del model2
    #     for param in model['model']:
    #         if 'decoder.embed_tokens.weight' in param:
    #             pass
    #         else:
    #             model['model'][param].div_(float(n))
    #     torch.save(model, bestensemble)
    #     del model
    #     bleu = bleufolder + 'BEAM4LEN1_bestmodel_ensemble{}_epoch{}_{}.out'.format(n, bestepoch - n + 1, bestepoch)
    #     print('evaluating {}'.format(bleu))
    #
    #     if 'IWSLTdeen' in modelname:
    #         if 'distill' in modelname:
    #             command = 'python generate.py data-bin/iwslt14_deen_jointdict_distill/ --gen-subset test --task translation_lev --path {} --beam 4 --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --iter-decode-force-max-iter --remove-bpe | tee {}'.format(
    #                 bestensemble, batch, bleu)
    #         else:
    #             command = 'python generate.py data-bin/iwslt14_deen_jointdict/ --gen-subset test --task translation_lev --path {} --beam 4 --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --iter-decode-force-max-iter --remove-bpe | tee {}'.format(
    #                 bestensemble, batch, bleu)
    #     if 'IWSLTende' in modelname:
    #         if 'distill' in modelname:
    #             command = 'python generate.py data-bin/iwslt14_ende_jointdict_distill/ --gen-subset test --task translation_lev --path {} --beam 4 --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --iter-decode-force-max-iter --remove-bpe | tee {}'.format(
    #                 bestensemble, batch, bleu)
    #         else:
    #             command = 'python generate.py data-bin/iwslt14_ende_jointdict/ --gen-subset test --task translation_lev --path {} --beam 4 --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --iter-decode-force-max-iter --remove-bpe | tee {}'.format(
    #                 bestensemble, batch, bleu)
    #
    #     os.system(command)
    #     command = './compound_split_bleu.sh {}'.format(bleu)
    #     os.system(command)
        #
        # bestbleu = 0
        # bestepoch = start
        # modelfolder = checkpointfolder + modelname + '/'
        # bleufolder = 'results/BLEU/' + modelname + '/ensemble{}_iter10_BEAM4LEN2/'.format(n)
        # os.system('mkdir -p {}'.format(bleufolder))
        #
        # jlist = [j for j in range(start, end + 1)]
        #
        # for j in jlist:
        #     cpname = modelfolder + 'checkpoint{}.pt'.format(j)
        #     model = torch.load(cpname)
        #     for i in range(1, n):
        #         cpname2 = modelfolder + 'checkpoint{}.pt'.format(j - i)
        #         model2 = torch.load(cpname2)
        #         for param in model['model']:
        #             if 'decoder.embed_tokens.weight' in param:
        #                 pass
        #             else:
        #                 model['model'][param].add_(model2['model'][param])
        #         del model2
        #     for param in model['model']:
        #         if 'decoder.embed_tokens.weight' in param:
        #             pass
        #         else:
        #             model['model'][param].div_(float(n))
        #     torch.save(model, ensemblemodelpath)
        #     del model
        #     bleu = bleufolder + 'checkpoint{}_ensemble{}.out'.format(j, n)
        #     print('evaluating {}'.format(bleu))
        #
        #     if 'IWSLTdeen' in modelname:
        #         if 'distill' in modelname:
        #             command = 'python generate.py data-bin/iwslt14_deen_jointdict_distill/ --gen-subset test --task translation_lev --path {} --beam 4 --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --remove-bpe --iter-decode-force-max-iter --iter-decode-with-beam 2 --quiet | tee {}'.format(
        #                 ensemblemodelpath, batch, bleu)
        #         else:
        #             command = 'python generate.py data-bin/iwslt14_deen_jointdict/ --gen-subset test --task translation_lev --path {} --beam 4 --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --remove-bpe --iter-decode-force-max-iter --iter-decode-with-beam 2 --quiet | tee {}'.format(
        #                 ensemblemodelpath, batch, bleu)
        #     elif 'IWSLTende' in modelname:
        #         if 'distill' in modelname:
        #             command = 'python generate.py data-bin/iwslt14_ende_jointdict_distill/ --gen-subset test --task translation_lev --path {} --beam 4 --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --remove-bpe --iter-decode-force-max-iter --iter-decode-with-beam 2 --quiet | tee {}'.format(
        #                 ensemblemodelpath, batch, bleu)
        #         else:
        #             command = 'python generate.py data-bin/iwslt14_ende_jointdict/ --gen-subset test --task translation_lev --path {} --beam 4 --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --remove-bpe --iter-decode-force-max-iter --iter-decode-with-beam 2 --quiet | tee {}'.format(
        #                 ensemblemodelpath, batch, bleu)
        #
        #     os.system(command)
        #     with open(bleu, 'r') as f:
        #         lines = f.read().splitlines()
        #         lastline = lines[-1].replace(',', '').split()
        #         if bestbleu < float(lastline[6]):
        #             bestbleu = float(lastline[6])
        #             bestepoch = j
        #         print('best bleu {} at epoch {}'.format(bestbleu, bestepoch))
        #
        # bestensemble = modelfolder + 'BEAM4LEN2_bestmodel_ensemble{}_epoch{}_{}.pt'.format(n, bestepoch - n + 1, bestepoch)
        # cpname = modelfolder + 'checkpoint{}.pt'.format(bestepoch)
        # model = torch.load(cpname)
        # for i in range(1, n):
        #     cpname2 = modelfolder + 'checkpoint{}.pt'.format(bestepoch - i)
        #     model2 = torch.load(cpname2)
        #     for param in model['model']:
        #         if 'decoder.embed_tokens.weight' in param:
        #             pass
        #         else:
        #             model['model'][param].add_(model2['model'][param])
        #     del model2
        # for param in model['model']:
        #     if 'decoder.embed_tokens.weight' in param:
        #         pass
        #     else:
        #         model['model'][param].div_(float(n))
        # torch.save(model, bestensemble)
        # del model
        # bleu = bleufolder + 'BEAM4LEN2_bestmodel_ensemble{}_epoch{}_{}.out'.format(n, bestepoch - n + 1, bestepoch)
        # print('evaluating {}'.format(bleu))
        #
        # if 'IWSLTdeen' in modelname:
        #     if 'distill' in modelname:
        #         command = 'python generate.py data-bin/iwslt14_deen_jointdict_distill/ --gen-subset test --task translation_lev --path {} --beam 4 --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --iter-decode-force-max-iter --iter-decode-with-beam 2 --remove-bpe | tee {}'.format(
        #             bestensemble, batch, bleu)
        #     else:
        #         command = 'python generate.py data-bin/iwslt14_deen_jointdict/ --gen-subset test --task translation_lev --path {} --beam 4 --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --iter-decode-force-max-iter --iter-decode-with-beam 2 --remove-bpe | tee {}'.format(
        #             bestensemble, batch, bleu)
        # if 'IWSLTende' in modelname:
        #     if 'distill' in modelname:
        #         command = 'python generate.py data-bin/iwslt14_ende_jointdict_distill/ --gen-subset test --task translation_lev --path {} --beam 4 --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --iter-decode-force-max-iter --iter-decode-with-beam 2 --remove-bpe | tee {}'.format(
        #             bestensemble, batch, bleu)
        #     else:
        #         command = 'python generate.py data-bin/iwslt14_ende_jointdict/ --gen-subset test --task translation_lev --path {} --beam 4 --batch-size {} --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --iter-decode-force-max-iter --iter-decode-with-beam 2 --remove-bpe | tee {}'.format(
        #             bestensemble, batch, bleu)
        # os.system(command)
        # command = './compound_split_bleu.sh {}'.format(bleu)
        # os.system(command)

