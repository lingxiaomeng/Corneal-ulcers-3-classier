from keras.engine.saving import load_model

import option
from data import DataLoader
import numpy as np

x_02_1 = [[9.70690131e-01, 2.93099042e-02]
    , [7.04955459e-01, 2.95044571e-01]
    , [2.00341672e-01, 7.99658358e-01]
    , [9.99855757e-01, 1.44236750e-04]
    , [9.93720174e-01, 6.27986575e-03]
    , [1.00000000e+00, 4.85936376e-08]
    , [1.91600487e-01, 8.08399498e-01]
    , [2.03694820e-01, 7.96305120e-01]
    , [6.81019068e-01, 3.18980873e-01]
    , [9.87509727e-01, 1.24901878e-02]
    , [9.76441264e-01, 2.35587638e-02]
    , [1.00000000e+00, 2.38540454e-09]
    , [9.99994874e-01, 5.10985774e-06]
    , [8.89597118e-01, 1.10402897e-01]
    , [2.24705085e-01, 7.75294900e-01]
    , [9.99224663e-01, 7.75286811e-04]
    , [9.91015255e-01, 8.98471102e-03]
    , [9.99221087e-01, 7.78844114e-04]
    , [7.71974504e-01, 2.28025496e-01]
    , [8.88238609e-01, 1.11761436e-01]
    , [2.07748845e-01, 7.92251170e-01]
    , [9.95410502e-01, 4.58945194e-03]
    , [9.88734543e-01, 1.12654306e-02]
    , [9.88260150e-01, 1.17398305e-02]
    , [9.99586642e-01, 4.13386035e-04]
    , [9.99999404e-01, 6.21125025e-07]
    , [7.35426620e-02, 9.26457345e-01]
    , [9.79381382e-01, 2.06186343e-02]
    , [9.53389585e-01, 4.66103777e-02]
    , [1.11882716e-01, 8.88117313e-01]
    , [4.11548465e-01, 5.88451564e-01]
    , [8.80672932e-02, 9.11932707e-01]
    , [9.99999881e-01, 6.82441765e-08]
    , [8.82048011e-01, 1.17952004e-01]
    , [4.55739230e-01, 5.44260800e-01]
    , [9.99875426e-01, 1.24569953e-04]
    , [3.10605139e-01, 6.89394832e-01]
    , [5.90843141e-01, 4.09156889e-01]
    , [3.12581778e-01, 6.87418222e-01]
    , [9.80381727e-01, 1.96182765e-02]
    , [9.99994755e-01, 5.20393587e-06]
    , [9.40868199e-01, 5.91317564e-02]
    , [7.92078599e-02, 9.20792162e-01]
    , [9.99999046e-01, 9.46488228e-07]
    , [9.99957085e-01, 4.29464999e-05]
    , [9.99238610e-01, 7.61436124e-04]
    , [9.99715745e-01, 2.84273672e-04]
    , [9.06274557e-01, 9.37253758e-02]
    , [9.92458284e-01, 7.54171098e-03]
    , [9.04895484e-01, 9.51045677e-02]
    , [9.97571409e-01, 2.42860336e-03]
    , [1.00000000e+00, 3.07952108e-09]
    , [9.99946237e-01, 5.37736487e-05]
    , [1.58154130e-01, 8.41845930e-01]
    , [9.99999881e-01, 7.31573877e-08]
    , [9.00330603e-01, 9.96694416e-02]
    , [9.99988675e-01, 1.13753940e-05]
    , [9.99958277e-01, 4.16930452e-05]
    , [9.99999046e-01, 1.00841976e-06]
    , [9.99183238e-01, 8.16717045e-04]
    , [9.99728262e-01, 2.71690544e-04]
    , [9.98132169e-01, 1.86787627e-03]
    , [3.35358083e-02, 9.66464221e-01]
    , [4.71261032e-02, 9.52873886e-01]
    , [9.88473475e-01, 1.15265362e-02]
    , [9.36371028e-01, 6.36290014e-02]
    , [1.00000000e+00, 4.28024300e-08]
    , [9.99978542e-01, 2.14340289e-05]
    , [3.65009680e-02, 9.63499010e-01]
    , [1.46855459e-01, 8.53144586e-01]
    , [7.98997432e-02, 9.20100212e-01]
    , [2.94684917e-01, 7.05315113e-01]
    , [9.97414589e-01, 2.58532795e-03]
    , [3.32761742e-02, 9.66723859e-01]
    , [9.99341905e-01, 6.58052100e-04]
    , [1.90651506e-01, 8.09348464e-01]
    , [9.95100439e-01, 4.89960192e-03]
    , [1.33053750e-01, 8.66946220e-01]
    , [9.99998093e-01, 1.92347170e-06]
    , [9.99854445e-01, 1.45515674e-04]
    , [9.99478638e-01, 5.21399372e-04]
    , [6.81610882e-01, 3.18389177e-01]
    , [6.93703234e-01, 3.06296796e-01]
    , [8.51171553e-01, 1.48828477e-01]
    , [9.98773873e-01, 1.22610910e-03]
    , [9.58512902e-01, 4.14870493e-02]
    , [9.99941707e-01, 5.82354805e-05]
    , [9.37953830e-01, 6.20461516e-02]
    , [9.97777045e-01, 2.22299341e-03]
    , [3.02844644e-01, 6.97155356e-01]
    , [9.99970078e-01, 2.98729974e-05]
    , [9.99999046e-01, 9.84149892e-07]
    , [8.89258146e-01, 1.10741861e-01]
    , [6.94407895e-02, 9.30559218e-01]
    , [2.52634343e-02, 9.74736512e-01]
    , [5.83156884e-01, 4.16843086e-01]
    , [9.99806345e-01, 1.93720829e-04]
    , [9.94277418e-01, 5.72266104e-03]
    , [9.89672184e-01, 1.03277816e-02]
    , [9.99805629e-01, 1.94346881e-04]
    , [9.99967933e-01, 3.20122017e-05]
    , [9.56482649e-01, 4.35174145e-02]
    , [9.90420461e-01, 9.57954302e-03]
    , [1.00000000e+00, 1.79314092e-08]
    , [9.87976372e-01, 1.20236073e-02]
    , [9.95734274e-01, 4.26572422e-03]
    , [1.62421972e-01, 8.37578058e-01]
    , [5.95018268e-02, 9.40498173e-01]
    , [7.22391754e-02, 9.27760780e-01]
    , [2.25989744e-01, 7.74010301e-01]
    , [7.61491880e-02, 9.23850834e-01]
    , [6.56578183e-01, 3.43421787e-01]
    , [9.99997497e-01, 2.44579019e-06]
    , [2.81269215e-02, 9.71873105e-01]
    , [4.85845469e-03, 9.95141506e-01]
    , [2.46135443e-02, 9.75386441e-01]
    , [9.58031178e-01, 4.19688523e-02]
    , [9.99991536e-01, 8.51804634e-06]
    , [9.64127898e-01, 3.58720422e-02]
    , [6.86018839e-02, 9.31398094e-01]
    , [9.99946713e-01, 5.33032799e-05]
    , [9.97086465e-01, 2.91359378e-03]
    , [4.25211310e-01, 5.74788630e-01]
    , [9.99862313e-01, 1.37616706e-04]
    , [9.97292817e-01, 2.70719337e-03]
    , [9.99982595e-01, 1.74458000e-05]
    , [9.99989390e-01, 1.06412117e-05]
    , [3.78849268e-01, 6.21150732e-01]
    , [6.77748919e-01, 3.22251141e-01]
    , [9.90084887e-01, 9.91514418e-03]
    , [1.70660198e-01, 8.29339802e-01]
    , [4.77918237e-01, 5.22081733e-01]
    , [9.78543341e-01, 2.14566700e-02]
    , [9.99976516e-01, 2.35084553e-05]
    , [6.52192757e-02, 9.34780717e-01]
    , [9.05790687e-01, 9.42093134e-02]
    , [9.99999523e-01, 4.40245856e-07]
    , [9.99996185e-01, 3.83684392e-06]
    , [9.64313328e-01, 3.56866568e-02]
    , [8.50162923e-01, 1.49837017e-01]
    , [9.99885082e-01, 1.14872018e-04]
    , [9.99999881e-01, 8.07753011e-08]
    , [9.94032085e-01, 5.96789317e-03]
    , [9.92385268e-01, 7.61472108e-03]
    , [1.15315765e-01, 8.84684265e-01]
    , [9.99997497e-01, 2.48246351e-06]
    , [1.00000000e+00, 3.05159586e-09]
    , [9.99756396e-01, 2.43564005e-04]
    , [2.25814208e-02, 9.77418602e-01]
    , [8.93146515e-01, 1.06853440e-01]
    , [9.98312950e-01, 1.68709690e-03]
    , [9.80011463e-01, 1.99885052e-02]
    , [9.75323379e-01, 2.46765874e-02]
    , [5.66782504e-02, 9.43321824e-01]
    , [9.99999762e-01, 2.02340274e-07]
    , [5.10600731e-02, 9.48939919e-01]
    , [5.12206927e-02, 9.48779345e-01]
    , [1.14676766e-01, 8.85323286e-01]
    , [1.74673628e-02, 9.82532620e-01]
    , [6.88070580e-02, 9.31192935e-01]
    , [4.31581698e-02, 9.56841767e-01]
    , [9.86581683e-01, 1.34183532e-02]
    , [9.99999046e-01, 9.76598244e-07]
    , [6.06486142e-01, 3.93513829e-01]
    , [5.43418005e-02, 9.45658267e-01]
    , [9.84506905e-01, 1.54931126e-02]
    , [8.65621567e-01, 1.34378389e-01]
    , [1.65446326e-02, 9.83455420e-01]
    , [9.98493075e-01, 1.50696270e-03]
    , [9.63519633e-01, 3.64803337e-02]
    , [4.25806791e-01, 5.74193239e-01]
    , [9.99562681e-01, 4.37357754e-04]
    , [9.98436034e-01, 1.56396441e-03]
    , [9.99051750e-01, 9.48245812e-04]
    , [1.00000000e+00, 3.61252452e-11]
    , [3.28967571e-02, 9.67103243e-01]
    , [5.65185070e-01, 4.34814930e-01]
    , [1.00000000e+00, 3.69656235e-08]
    , [4.15531933e-01, 5.84468067e-01]
    , [1.60066523e-02, 9.83993292e-01]
    , [3.09267100e-02, 9.69073236e-01]
    , [9.99970436e-01, 2.96075923e-05]
    , [9.15089488e-01, 8.49104896e-02]
    , [1.00000000e+00, 2.11917293e-08]
    , [9.99998093e-01, 1.95367625e-06]
    , [9.96231973e-01, 3.76799027e-03]
    , [7.53566444e-01, 2.46433571e-01]
    , [9.99472678e-01, 5.27310011e-04]
    , [8.20844024e-02, 9.17915642e-01]
    , [1.30798146e-01, 8.69201839e-01]
    , [9.99530911e-01, 4.69080434e-04]
    , [9.99735534e-01, 2.64414004e-04]
    , [9.99999881e-01, 6.21288549e-08]
    , [2.46138513e-01, 7.53861487e-01]
    , [4.22857739e-02, 9.57714200e-01]
    , [9.99994040e-01, 5.94595167e-06]
    , [9.97437835e-01, 2.56218063e-03]
    , [1.46262906e-02, 9.85373735e-01]
    , [3.96767706e-01, 6.03232265e-01]
    , [9.20600057e-01, 7.93999359e-02]
    , [9.56635773e-01, 4.33642045e-02]
    , [8.40648264e-02, 9.15935218e-01]
    , [3.33868014e-03, 9.96661305e-01]
    , [2.30577923e-02, 9.76942182e-01]
    , [1.00000000e+00, 1.28111148e-08]
    , [8.83230686e-01, 1.16769321e-01]
    , [9.73539233e-01, 2.64607184e-02]
    , [9.99412894e-01, 5.87059883e-04]
    , [5.23882449e-01, 4.76117522e-01]
    , [9.98048902e-01, 1.95106550e-03]
    , [9.97899175e-01, 2.10083928e-03]
    , [9.86551404e-01, 1.34485662e-02]
    , [9.99195039e-01, 8.04984476e-04]
    , [1.00000000e+00, 7.04964576e-09]]

x_12_0 = [[9.92054224e-01, 7.94584677e-03]
    , [1.46073010e-03, 9.98539329e-01]
    , [9.99571025e-01, 4.28976928e-04]
    , [2.46615917e-03, 9.97533798e-01]
    , [9.99330401e-01, 6.69614936e-04]
    , [3.17361514e-06, 9.99996781e-01]
    , [9.99456584e-01, 5.43459610e-04]
    , [9.97943103e-01, 2.05692439e-03]
    , [9.95404720e-01, 4.59522894e-03]
    , [9.85957384e-01, 1.40426168e-02]
    , [1.11991796e-03, 9.98880088e-01]
    , [4.50543030e-06, 9.99995470e-01]
    , [7.29510657e-06, 9.99992728e-01]
    , [9.94428992e-01, 5.57107152e-03]
    , [9.98897910e-01, 1.10211968e-03]
    , [1.52568173e-04, 9.99847412e-01]
    , [9.49484050e-01, 5.05159050e-02]
    , [8.44996885e-06, 9.99991536e-01]
    , [9.99594629e-01, 4.05367377e-04]
    , [9.98814464e-01, 1.18561077e-03]
    , [9.95143056e-01, 4.85701673e-03]
    , [2.10553184e-02, 9.78944659e-01]
    , [5.09946585e-01, 4.90053385e-01]
    , [1.38027928e-04, 9.99861956e-01]
    , [1.10370427e-04, 9.99889612e-01]
    , [1.16310657e-05, 9.99988317e-01]
    , [9.95850682e-01, 4.14931821e-03]
    , [9.64151382e-01, 3.58485579e-02]
    , [9.16563570e-01, 8.34364146e-02]
    , [9.99261558e-01, 7.38392293e-04]
    , [9.88341391e-01, 1.16586192e-02]
    , [9.97420430e-01, 2.57956400e-03]
    , [8.86908529e-05, 9.99911308e-01]
    , [9.94952440e-01, 5.04752761e-03]
    , [9.95508313e-01, 4.49164351e-03]
    , [4.34149442e-05, 9.99956608e-01]
    , [9.97206271e-01, 2.79366435e-03]
    , [9.85091388e-01, 1.49085913e-02]
    , [7.81660140e-01, 2.18339846e-01]
    , [9.89973307e-01, 1.00267446e-02]
    , [3.84392406e-05, 9.99961615e-01]
    , [9.18528974e-01, 8.14710110e-02]
    , [9.93827760e-01, 6.17225235e-03]
    , [4.61165700e-06, 9.99995351e-01]
    , [3.86684042e-06, 9.99996185e-01]
    , [3.50690266e-06, 9.99996543e-01]
    , [2.22245071e-04, 9.99777734e-01]
    , [9.96989191e-01, 3.01086879e-03]
    , [6.67802698e-04, 9.99332130e-01]
    , [1.61323491e-02, 9.83867645e-01]
    , [1.90120568e-06, 9.99998093e-01]
    , [8.21501726e-06, 9.99991775e-01]
    , [3.61537299e-04, 9.99638438e-01]
    , [9.98035848e-01, 1.96412671e-03]
    , [7.18801675e-06, 9.99992847e-01]
    , [9.94819343e-01, 5.18071791e-03]
    , [3.58245961e-06, 9.99996424e-01]
    , [2.35463049e-05, 9.99976397e-01]
    , [7.76823128e-07, 9.99999166e-01]
    , [1.31495675e-04, 9.99868512e-01]
    , [1.38647767e-04, 9.99861360e-01]
    , [2.98988163e-01, 7.01011837e-01]
    , [9.95925903e-01, 4.07415628e-03]
    , [9.98907685e-01, 1.09235232e-03]
    , [1.51264481e-04, 9.99848723e-01]
    , [9.91494298e-01, 8.50572716e-03]
    , [1.43596446e-06, 9.99998569e-01]
    , [6.32495736e-04, 9.99367535e-01]
    , [9.98323262e-01, 1.67674827e-03]
    , [9.54363585e-01, 4.56363596e-02]
    , [9.99065816e-01, 9.34152224e-04]
    , [9.99136269e-01, 8.63775669e-04]
    , [5.39893517e-04, 9.99460161e-01]
    , [9.99340236e-01, 6.59711077e-04]
    , [2.19882291e-04, 9.99780118e-01]
    , [9.98408258e-01, 1.59174565e-03]
    , [9.52080905e-01, 4.79190610e-02]
    , [9.98661757e-01, 1.33826677e-03]
    , [1.84504722e-06, 9.99998212e-01]
    , [6.67950153e-05, 9.99933243e-01]
    , [2.44416809e-03, 9.97555852e-01]
    , [9.88150358e-01, 1.18496194e-02]
    , [9.95250225e-01, 4.74980334e-03]
    , [9.98019934e-01, 1.98004558e-03]
    , [9.99377589e-05, 9.99900103e-01]
    , [8.18215404e-03, 9.91817832e-01]
    , [2.36309905e-04, 9.99763668e-01]
    , [9.97903228e-01, 2.09679245e-03]
    , [3.89800734e-05, 9.99961019e-01]
    , [9.97659922e-01, 2.34010699e-03]
    , [1.41477445e-04, 9.99858499e-01]
    , [7.14527278e-06, 9.99992847e-01]
    , [9.94120777e-01, 5.87921776e-03]
    , [9.99110758e-01, 8.89258576e-04]
    , [9.97893155e-01, 2.10684282e-03]
    , [9.97079492e-01, 2.92050559e-03]
    , [8.60189321e-05, 9.99913931e-01]
    , [1.46991137e-04, 9.99853015e-01]
    , [6.69173431e-04, 9.99330878e-01]
    , [9.96023536e-01, 3.97646241e-03]
    , [1.35220721e-01, 8.64779294e-01]
    , [6.87729120e-01, 3.12270880e-01]
    , [9.96605039e-01, 3.39491293e-03]
    , [1.39515651e-05, 9.99986053e-01]
    , [8.67292225e-01, 1.32707819e-01]
    , [2.60856996e-05, 9.99973893e-01]
    , [9.90841866e-01, 9.15812794e-03]
    , [9.98652935e-01, 1.34705554e-03]
    , [9.93721247e-01, 6.27877144e-03]
    , [9.99362648e-01, 6.37371559e-04]
    , [9.96696472e-01, 3.30354949e-03]
    , [9.97138381e-01, 2.86158524e-03]
    , [7.81822109e-06, 9.99992132e-01]
    , [9.94392633e-01, 5.60728088e-03]
    , [9.99743760e-01, 2.56271887e-04]
    , [9.99359906e-01, 6.40150683e-04]
    , [5.65518683e-04, 9.99434412e-01]
    , [1.23570673e-03, 9.98764277e-01]
    , [3.20946798e-02, 9.67905343e-01]
    , [9.95331347e-01, 4.66864882e-03]
    , [1.16335257e-04, 9.99883652e-01]
    , [1.97162663e-04, 9.99802887e-01]
    , [9.94474709e-01, 5.52529655e-03]
    , [2.80348086e-05, 9.99971986e-01]
    , [9.53392358e-04, 9.99046624e-01]
    , [8.75622392e-01, 1.24377556e-01]
    , [7.88143778e-04, 9.99211788e-01]
    , [9.99779522e-01, 2.20527494e-04]
    , [9.96655345e-01, 3.34460055e-03]
    , [9.95430231e-01, 4.56978474e-03]
    , [9.96499062e-01, 3.50087555e-03]
    , [9.84740674e-01, 1.52593004e-02]
    , [1.37133594e-03, 9.98628736e-01]
    , [1.42392601e-05, 9.99985814e-01]
    , [9.98204708e-01, 1.79529947e-03]
    , [9.95664656e-01, 4.33541089e-03]
    , [4.05379342e-06, 9.99995947e-01]
    , [2.72328005e-04, 9.99727666e-01]
    , [9.98325288e-01, 1.67475245e-03]
    , [9.92285252e-01, 7.71478796e-03]
    , [1.11992798e-04, 9.99887943e-01]
    , [2.14349293e-05, 9.99978542e-01]
    , [9.54380333e-01, 4.56196703e-02]
    , [9.89501417e-01, 1.04985377e-02]
    , [9.96232331e-01, 3.76767898e-03]
    , [1.63204058e-05, 9.99983668e-01]
    , [1.56203396e-06, 9.99998450e-01]
    , [1.01252102e-04, 9.99898791e-01]
    , [9.98737395e-01, 1.26266736e-03]
    , [9.93299603e-01, 6.70037093e-03]
    , [1.80175179e-03, 9.98198330e-01]
    , [1.17429961e-02, 9.88256991e-01]
    , [9.88815129e-01, 1.11848684e-02]
    , [9.94388402e-01, 5.61154727e-03]
    , [4.11513338e-06, 9.99995828e-01]
    , [9.99920607e-01, 7.93792933e-05]
    , [9.02907193e-01, 9.70927849e-02]
    , [8.79677236e-01, 1.20322779e-01]
    , [9.97219324e-01, 2.78070290e-03]
    , [9.94850338e-01, 5.14970534e-03]
    , [9.98227417e-01, 1.77257205e-03]
    , [1.24891013e-01, 8.75109017e-01]
    , [1.60744617e-06, 9.99998450e-01]
    , [9.88389373e-01, 1.16106244e-02]
    , [9.99332011e-01, 6.67950779e-04]
    , [2.68252043e-04, 9.99731719e-01]
    , [9.57321465e-01, 4.26784754e-02]
    , [9.99430954e-01, 5.69060387e-04]
    , [9.89077866e-01, 1.09220529e-02]
    , [9.99386072e-01, 6.13875454e-04]
    , [9.98869240e-01, 1.13070838e-03]
    , [4.34975147e-01, 5.65024853e-01]
    , [1.31024804e-04, 9.99868989e-01]
    , [3.66041627e-06, 9.99996305e-01]
    , [5.76802404e-06, 9.99994278e-01]
    , [9.97286797e-01, 2.71324278e-03]
    , [1.37357138e-05, 9.99986291e-01]
    , [2.07932408e-06, 9.99997973e-01]
    , [9.99513507e-01, 4.86439414e-04]
    , [9.96285558e-01, 3.71447741e-03]
    , [9.97222781e-01, 2.77728448e-03]
    , [8.41941946e-05, 9.99915838e-01]
    , [9.95365620e-01, 4.63433936e-03]
    , [4.85177297e-05, 9.99951482e-01]
    , [2.04129192e-05, 9.99979615e-01]
    , [7.29388511e-03, 9.92706120e-01]
    , [9.93403137e-01, 6.59689261e-03]
    , [6.63039509e-06, 9.99993324e-01]
    , [9.95455146e-01, 4.54485603e-03]
    , [9.88831162e-01, 1.11687742e-02]
    , [7.99070229e-04, 9.99201000e-01]
    , [1.26549603e-05, 9.99987364e-01]
    , [1.03558741e-05, 9.99989629e-01]
    , [9.97961879e-01, 2.03808094e-03]
    , [9.98005211e-01, 1.99481123e-03]
    , [6.96994102e-05, 9.99930263e-01]
    , [3.28842143e-05, 9.99967098e-01]
    , [9.97887313e-01, 2.11268757e-03]
    , [6.48315996e-02, 9.35168386e-01]
    , [9.91268873e-01, 8.73105600e-03]
    , [9.93573248e-01, 6.42672321e-03]
    , [9.98681009e-01, 1.31899701e-03]
    , [9.99545991e-01, 4.53924120e-04]
    , [9.98224795e-01, 1.77518604e-03]
    , [4.15187287e-06, 9.99995828e-01]
    , [9.99387145e-01, 6.12848089e-04]
    , [8.13241363e-01, 1.86758682e-01]
    , [8.67872965e-04, 9.99132097e-01]
    , [9.99139190e-01, 8.60791537e-04]
    , [1.92090622e-04, 9.99807894e-01]
    , [2.05802426e-04, 9.99794185e-01]
    , [5.55435956e-01, 4.44564074e-01]
    , [2.17690904e-04, 9.99782264e-01]
    , [4.86075396e-06, 9.99995112e-01]]
#
# print(x_0_12)
# print(x_02_1)

# x_0_12:  0[1 0]     1[0 1]
#             点     点片+片
#             01       2
# x_02_1:  0[1 0]     1[0 1]
#            点+片    点片
#             02       1

data = DataLoader(option.args)

im, real_label, filename = data.get_test()

x_02_1 = [[9.67133403e-01, 3.28665674e-02]
    , [9.99986172e-01, 1.38755622e-05]
    , [8.00456703e-02, 9.19954300e-01]
    , [9.99998569e-01, 1.41293492e-06]
    , [9.99715626e-01, 2.84349808e-04]
    , [1.00000000e+00, 7.67841524e-09]
    , [6.09025230e-09, 1.00000000e+00]
    , [1.49642972e-06, 9.99998450e-01]
    , [1.55078905e-09, 1.00000000e+00]
    , [6.85642287e-02, 9.31435704e-01]
    , [6.52312100e-01, 3.47687930e-01]
    , [1.00000000e+00, 9.19431375e-09]
    , [9.99996066e-01, 3.90800187e-06]
    , [3.47854465e-01, 6.52145505e-01]
    , [8.08860525e-04, 9.99191105e-01]
    , [9.99995589e-01, 4.41176553e-06]
    , [9.99987602e-01, 1.23561067e-05]
    , [9.99989629e-01, 1.03457551e-05]
    , [6.46002150e-07, 9.99999404e-01]
    , [8.56729627e-01, 1.43270418e-01]
    , [2.73577143e-07, 9.99999821e-01]
    , [9.99990940e-01, 9.06977766e-06]
    , [8.99515033e-01, 1.00484908e-01]
    , [9.99544501e-01, 4.55421337e-04]
    , [9.97112095e-01, 2.88788718e-03]
    , [9.99999821e-01, 2.88627575e-07]
    , [3.39075405e-06, 9.99996662e-01]
    , [9.99980569e-01, 1.94724234e-05]
    , [9.99750435e-01, 2.49565433e-04]
    , [2.80431216e-08, 1.00000000e+00]
    , [9.96734440e-01, 3.26565607e-03]
    , [3.07206040e-12, 1.00000000e+00]
    , [9.99995589e-01, 4.42271403e-06]
    , [9.02737796e-01, 9.72621590e-02]
    , [1.02684580e-05, 9.99989688e-01]
    , [9.99999881e-01, 1.66464716e-07]
    , [3.69241595e-01, 6.30758405e-01]
    , [4.33272216e-05, 9.99956727e-01]
    , [9.36568260e-01, 6.34317547e-02]
    , [9.99971867e-01, 2.81311914e-05]
    , [9.96131897e-01, 3.86813423e-03]
    , [9.35226753e-02, 9.06477273e-01]
    , [3.31320721e-10, 1.00000000e+00]
    , [9.99999642e-01, 3.70627731e-07]
    , [9.99992013e-01, 7.95645519e-06]
    , [9.99985576e-01, 1.44588721e-05]
    , [9.99907136e-01, 9.28693044e-05]
    , [3.03039155e-06, 9.99997020e-01]
    , [9.45915580e-01, 5.40843792e-02]
    , [9.99998093e-01, 1.91502272e-06]
    , [1.00000000e+00, 1.32179849e-08]
    , [9.99999881e-01, 1.63107273e-07]
    , [9.99985516e-01, 1.45430340e-05]
    , [1.78787013e-06, 9.99998212e-01]
    , [1.00000000e+00, 2.89096769e-09]
    , [8.23199855e-07, 9.99999166e-01]
    , [1.00000000e+00, 2.57932586e-09]
    , [9.99997616e-01, 2.39552901e-06]
    , [1.00000000e+00, 4.27555680e-09]
    , [9.99999881e-01, 1.30798654e-07]
    , [9.99996662e-01, 3.29844852e-06]
    , [9.99978840e-01, 2.11972547e-05]
    , [1.12005183e-03, 9.98880029e-01]
    , [3.87014094e-04, 9.99612987e-01]
    , [9.99997139e-01, 2.87539865e-06]
    , [9.94894207e-01, 5.10578789e-03]
    , [1.00000000e+00, 5.89531313e-09]
    , [9.99966025e-01, 3.40185543e-05]
    , [4.50687231e-12, 1.00000000e+00]
    , [1.50809001e-06, 9.99998450e-01]
    , [1.76773982e-12, 1.00000000e+00]
    , [2.08635069e-02, 9.79136467e-01]
    , [9.99994874e-01, 5.07560799e-06]
    , [5.41224509e-14, 1.00000000e+00]
    , [9.99957860e-01, 4.22223202e-05]
    , [1.05360614e-05, 9.99989510e-01]
    , [9.99999821e-01, 2.22273087e-07]
    , [9.30849671e-01, 6.91502690e-02]
    , [9.99999821e-01, 2.05579198e-07]
    , [9.99784350e-01, 2.15598848e-04]
    , [9.99983788e-01, 1.62411143e-05]
    , [5.48512954e-03, 9.94514942e-01]
    , [6.26081373e-08, 9.99999881e-01]
    , [1.22925081e-09, 1.00000000e+00]
    , [9.99983788e-01, 1.62414253e-05]
    , [3.22678573e-02, 9.67732131e-01]
    , [9.99993443e-01, 6.54118548e-06]
    , [1.13576343e-02, 9.88642395e-01]
    , [9.99997616e-01, 2.33320452e-06]
    , [2.37895511e-02, 9.76210475e-01]
    , [9.99996543e-01, 3.49316406e-06]
    , [9.99998569e-01, 1.38054804e-06]
    , [1.12141549e-07, 9.99999881e-01]
    , [7.25655369e-11, 1.00000000e+00]
    , [1.40600960e-07, 9.99999881e-01]
    , [3.16297961e-03, 9.96837020e-01]
    , [9.99980927e-01, 1.91068757e-05]
    , [9.99994397e-01, 5.57690237e-06]
    , [9.99958217e-01, 4.17476331e-05]
    , [9.99997735e-01, 2.26034558e-06]
    , [9.27168578e-02, 9.07283187e-01]
    , [2.48683733e-04, 9.99751270e-01]
    , [9.99999523e-01, 4.50402837e-07]
    , [9.99996960e-01, 3.04188052e-06]
    , [8.74747753e-01, 1.25252277e-01]
    , [9.99988377e-01, 1.17338423e-05]
    , [9.99991536e-01, 8.42480040e-06]
    , [1.67654813e-04, 9.99832273e-01]
    , [8.74220518e-09, 1.00000000e+00]
    , [1.29236666e-08, 1.00000000e+00]
    , [1.71024150e-11, 1.00000000e+00]
    , [8.03157775e-07, 9.99999166e-01]
    , [9.99998093e-01, 1.86138254e-06]
    , [1.74334106e-11, 1.00000000e+00]
    , [1.45671175e-09, 1.00000000e+00]
    , [2.66359795e-10, 1.00000000e+00]
    , [9.99999881e-01, 1.58336888e-07]
    , [9.99999225e-01, 6.57202236e-07]
    , [9.99931455e-01, 6.85425039e-05]
    , [7.96108335e-09, 1.00000000e+00]
    , [9.99894738e-01, 1.05194966e-04]
    , [9.99996960e-01, 3.06818333e-06]
    , [9.99832749e-01, 1.67269362e-04]
    , [9.82388616e-01, 1.76113993e-02]
    , [9.99851584e-01, 1.48439081e-04]
    , [9.99991775e-01, 8.21389676e-06]
    , [9.99949098e-01, 5.09065758e-05]
    , [2.03651097e-02, 9.79634941e-01]
    , [2.33994216e-01, 7.66005754e-01]
    , [9.80705559e-01, 1.92944296e-02]
    , [9.99998927e-01, 1.12147904e-06]
    , [1.55871138e-01, 8.44128788e-01]
    , [9.99986529e-01, 1.34446582e-05]
    , [9.99987483e-01, 1.25262204e-05]
    , [1.97712370e-06, 9.99997973e-01]
    , [4.12437972e-03, 9.95875657e-01]
    , [9.99998093e-01, 1.92992457e-06]
    , [9.99216676e-01, 7.83345487e-04]
    , [2.03157333e-06, 9.99997973e-01]
    , [9.99984384e-01, 1.56595652e-05]
    , [9.99998212e-01, 1.80476820e-06]
    , [9.99997735e-01, 2.31620265e-06]
    , [9.99989152e-01, 1.08396280e-05]
    , [9.99999523e-01, 4.64779021e-07]
    , [9.99877930e-01, 1.22093305e-04]
    , [9.99991536e-01, 8.47901356e-06]
    , [1.00000000e+00, 1.08995346e-09]
    , [9.99999166e-01, 7.85096006e-07]
    , [7.94636890e-07, 9.99999166e-01]
    , [4.15627100e-02, 9.58437264e-01]
    , [9.95667279e-01, 4.33277898e-03]
    , [9.99999881e-01, 1.05502480e-07]
    , [7.20567167e-01, 2.79432803e-01]
    , [3.92041011e-06, 9.99996066e-01]
    , [9.99999881e-01, 1.55726397e-07]
    , [7.80970464e-03, 9.92190242e-01]
    , [6.51379324e-08, 9.99999881e-01]
    , [9.99966621e-01, 3.33312164e-05]
    , [2.31972997e-13, 1.00000000e+00]
    , [3.77345495e-02, 9.62265491e-01]
    , [1.61017866e-11, 1.00000000e+00]
    , [9.99996006e-01, 4.08604728e-06]
    , [9.99978542e-01, 2.14236479e-05]
    , [9.05900300e-01, 9.40996408e-02]
    , [2.59684647e-08, 1.00000000e+00]
    , [9.99835014e-01, 1.64915662e-04]
    , [1.71299443e-01, 8.28700662e-01]
    , [2.07620339e-08, 1.00000000e+00]
    , [9.95525658e-01, 4.47426783e-03]
    , [9.99983430e-01, 1.65988477e-05]
    , [8.94808109e-05, 9.99910474e-01]
    , [9.01881099e-01, 9.81188938e-02]
    , [9.99998093e-01, 1.90819537e-06]
    , [9.99951541e-01, 4.83930162e-05]
    , [9.99996662e-01, 3.29773093e-06]
    , [4.24451412e-11, 1.00000000e+00]
    , [9.99949634e-01, 5.02787989e-05]
    , [9.99999046e-01, 9.79184279e-07]
    , [5.21168634e-02, 9.47883010e-01]
    , [4.03418055e-09, 1.00000000e+00]
    , [9.27487687e-09, 1.00000000e+00]
    , [9.99998868e-01, 1.22740028e-06]
    , [2.69175172e-01, 7.30824888e-01]
    , [9.99960780e-01, 3.92721195e-05]
    , [9.99999821e-01, 2.79986068e-07]
    , [9.45992112e-01, 5.40079996e-02]
    , [5.46200283e-07, 9.99999404e-01]
    , [9.99997973e-01, 2.00459272e-06]
    , [1.22647872e-02, 9.87735152e-01]
    , [2.12483769e-07, 9.99999821e-01]
    , [9.99999881e-01, 1.31164029e-07]
    , [9.99987006e-01, 1.29518858e-05]
    , [9.99982834e-01, 1.71267857e-05]
    , [5.59639093e-03, 9.94403660e-01]
    , [1.37466031e-05, 9.99986291e-01]
    , [9.99983668e-01, 1.63747700e-05]
    , [1.00000000e+00, 4.37668426e-08]
    , [1.68779017e-08, 1.00000000e+00]
    , [9.99648571e-01, 3.51388357e-04]
    , [3.60626400e-01, 6.39373541e-01]
    , [9.73303199e-01, 2.66967155e-02]
    , [1.24786348e-09, 1.00000000e+00]
    , [9.25064114e-08, 9.99999881e-01]
    , [3.08040902e-03, 9.96919632e-01]
    , [9.99998271e-01, 1.62555966e-06]
    , [1.48210693e-02, 9.85178828e-01]
    , [1.13585386e-02, 9.88641441e-01]
    , [9.98353481e-01, 1.64650637e-03]
    , [3.72143835e-02, 9.62785602e-01]
    , [9.72692668e-01, 2.73072962e-02]
    , [9.99991894e-01, 8.07570996e-06]
    , [9.99926746e-01, 7.32336775e-05]
    , [9.99997497e-01, 2.47627077e-06]
    , [9.99991536e-01, 8.45867635e-06]]

x00 = 0
x012 = 0
x002 = 0
x01 = 0

x10 = 0
x112 = 0
x102 = 0
x11 = 0

x20 = 0
x212 = 0
x202 = 0
x21 = 0

for i in range(len(real_label)):
    if real_label[i] == 0:
        # print(str(x_12_0[i]) + " " + str(x_02_1[i]))
        if x_12_0[i][1] > 0.5:
            x00 += 1
        else:
            x012 += 1

        if x_02_1[i][0] > 0.6:
            x002 += 1
        else:
            x01 += 1

print()

for i in range(len(real_label)):
    if real_label[i] == 1:
        # print(str(x_12_0[i]) + " " + str(x_02_1[i]))
        if x_12_0[i][1] > 0.5:
            x10 += 1
        else:
            x112 += 1

        if x_02_1[i][0] > 0.6:
            x102 += 1
        else:
            x11 += 1

print()

for i in range(len(real_label)):
    if real_label[i] == 2:
        # print(str(x_12_0[i]) + " " + str(x_02_1[i]))
        if x_12_0[i][1] > 0.5:
            x20 += 1
        else:
            x212 += 1

        if x_02_1[i][0] > 0.6:
            x202 += 1
        else:
            x21 += 1

print('{} {}'.format(x00, x012))
print('{} {}'.format(x10, x112))
print('{} {}'.format(x20, x212))
print()
print('{} {}'.format(x002, x01))
print('{} {}'.format(x102, x11))
print('{} {}'.format(x202, x21))
print()
x = np.empty((len(real_label), 3))

for i in range(len(real_label)):
    x[i][0] = x_02_1[i][0] + x_12_0[i][1]
    x[i][1] = x_02_1[i][1] + x_12_0[i][0]
    x[i][2] = x_02_1[i][0] + x_12_0[i][0]


def maxindex(sublist):
    if sublist[0] > sublist[1] and sublist[0] > sublist[2]:
        return 0
    if sublist[1] > sublist[0] and sublist[1] > sublist[2]:
        return 1
    if sublist[2] > sublist[0] and sublist[2] > sublist[1]:
        return 2


x00 = 0
x01 = 0
x02 = 0
x10 = 0
x11 = 0
x12 = 0
x20 = 0
x21 = 0
x22 = 0

i = 0
for i in range(len(x)):
    if real_label[i] == 0:
        index = maxindex(x[i])
        if index == 0:
            x00 += 1
        if index == 1:
            x01 += 1
        if index == 2:
            x02 += 1
    if real_label[i] == 1:
        index = maxindex(x[i])
        if index == 0:
            x10 += 1
        if index == 1:
            x11 += 1
        if index == 2:
            x12 += 1

    if real_label[i] == 2:
        index = maxindex(x[i])
        if index == 0:
            x20 += 1
        if index == 1:
            x21 += 1
        if index == 2:
            x22 += 1

print(x)

print("{} {} {}".format(x00, x01, x02))
print("{} {} {}".format(x10, x11, x12))
print("{} {} {}".format(x20, x21, x22))
