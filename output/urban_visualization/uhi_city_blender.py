# Blender Python Script for UHI Hotspot Visualization
# Run this script in Blender (Text Editor > Run Script)
# Requires Blender 2.8+

import bpy
import bmesh
import math
from mathutils import Vector

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Hotspot data
hotspot_data = [{'position': {'x': 0.0, 'y': 0.0, 'z': 32.2030691228251}, 'uhi_value': 0.03171157540733027, 'intensity': 0.6440613824565019, 'color': {'r': np.float64(0.5762455298260076), 'g': 1, 'b': 0}}, {'position': {'x': 0.0, 'y': 1.1111111111112848, 'z': 48.5891635175697}, 'uhi_value': 0.11148881098461108, 'intensity': 0.971783270351394, 'color': {'r': 1, 'g': np.float64(0.11286691859442399), 'b': 0}}, {'position': {'x': 0.0, 'y': 2.2222222222225696, 'z': 46.06315594629452}, 'uhi_value': 0.09919070649253356, 'intensity': 0.9212631189258904, 'color': {'r': 1, 'g': np.float64(0.31494752429643835), 'b': 0}}, {'position': {'x': 0.0, 'y': 3.333333333333144, 'z': 27.106932158816548}, 'uhi_value': 0.006900555202162288, 'intensity': 0.542138643176331, 'color': {'r': np.float64(0.16855457270532392), 'g': 1, 'b': 0}}, {'position': {'x': 0.0, 'y': 4.444444444444429, 'z': 11.655184744804767}, 'uhi_value': -0.06832772457166911, 'intensity': 0.23310369489609534, 'color': {'r': 0, 'g': np.float64(0.9324147795843813), 'b': 1}}, {'position': {'x': 0.0, 'y': 5.5555555555557135, 'z': 38.97211755308244}, 'uhi_value': 0.0646673218009223, 'intensity': 0.7794423510616488, 'color': {'r': 1, 'g': np.float64(0.8822305957534047), 'b': 0}}, {'position': {'x': 0.0, 'y': 6.666666666666998, 'z': 20.535027313480114}, 'uhi_value': -0.02509537918022655, 'intensity': 0.4107005462696023, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.35719781492159086)}}, {'position': {'x': 0.0, 'y': 7.7777777777775725, 'z': 36.63992026048594}, 'uhi_value': 0.053312800806350195, 'intensity': 0.7327984052097187, 'color': {'r': np.float64(0.9311936208388749), 'g': 1, 'b': 0}}, {'position': {'x': 0.0, 'y': 8.888888888888857, 'z': 50.0}, 'uhi_value': 0.11835760056365577, 'intensity': 1.0, 'color': {'r': 1, 'g': 0.0, 'b': 0}}, {'position': {'x': 0.0, 'y': 10.000000000000142, 'z': 41.827982308206494}, 'uhi_value': 0.07857136689292765, 'intensity': 0.8365596461641299, 'color': {'r': 1, 'g': np.float64(0.6537614153434803), 'b': 0}}, {'position': {'x': 1.1111111111105743, 'y': 0.0, 'z': 15.661847120736144}, 'uhi_value': -0.04882091345834968, 'intensity': 0.3132369424147229, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.7470522303411085)}}, {'position': {'x': 1.1111111111105743, 'y': 1.1111111111112848, 'z': 46.349210196338106}, 'uhi_value': 0.10058338840585934, 'intensity': 0.9269842039267622, 'color': {'r': 1, 'g': np.float64(0.2920631842929513), 'b': 0}}, {'position': {'x': 1.1111111111105743, 'y': 2.2222222222225696, 'z': 21.391655550288462}, 'uhi_value': -0.020924804362109528, 'intensity': 0.42783311100576926, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.28866755597692295)}}, {'position': {'x': 1.1111111111105743, 'y': 3.333333333333144, 'z': 29.64968728052838}, 'uhi_value': 0.0192801967124176, 'intensity': 0.5929937456105676, 'color': {'r': np.float64(0.37197498244227045), 'g': 1, 'b': 0}}, {'position': {'x': 1.1111111111105743, 'y': 4.444444444444429, 'z': 15.134907943523917}, 'uhi_value': -0.05138636620521294, 'intensity': 0.30269815887047835, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.7892073645180866)}}, {'position': {'x': 1.1111111111105743, 'y': 5.5555555555557135, 'z': 17.318861325660396}, 'uhi_value': -0.04075358457590778, 'intensity': 0.3463772265132079, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.6144910939471684)}}, {'position': {'x': 1.1111111111105743, 'y': 6.666666666666998, 'z': 20.038609176976532}, 'uhi_value': -0.027512237381249884, 'intensity': 0.40077218353953065, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.3969112658418774)}}, {'position': {'x': 1.1111111111105743, 'y': 7.7777777777775725, 'z': 31.328701367457995}, 'uhi_value': 0.027454634081307442, 'intensity': 0.6265740273491599, 'color': {'r': np.float64(0.5062961093966396), 'g': 1, 'b': 0}}, {'position': {'x': 1.1111111111105743, 'y': 8.888888888888857, 'z': 13.50646142202592}, 'uhi_value': -0.05931461064473085, 'intensity': 0.2701292284405184, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.9194830862379264)}}, {'position': {'x': 1.1111111111105743, 'y': 10.000000000000142, 'z': 21.252750956796504}, 'uhi_value': -0.02160107438782894, 'intensity': 0.4250550191359301, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.2997799234562797)}}, {'position': {'x': 2.2222222222225696, 'y': 0.0, 'z': 20.55808573881477}, 'uhi_value': -0.02498311707641774, 'intensity': 0.41116171477629543, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.3553531408948183)}}, {'position': {'x': 2.2222222222225696, 'y': 1.1111111111112848, 'z': 14.874547515435568}, 'uhi_value': -0.052653955339403924, 'intensity': 0.29749095030871137, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.8100361987651545)}}, {'position': {'x': 2.2222222222225696, 'y': 2.2222222222225696, 'z': 34.94794433595757}, 'uhi_value': 0.045075257516871914, 'intensity': 0.6989588867191515, 'color': {'r': np.float64(0.7958355468766061), 'g': 1, 'b': 0}}, {'position': {'x': 2.2222222222225696, 'y': 3.333333333333144, 'z': 36.000774846370334}, 'uhi_value': 0.05020106148386756, 'intensity': 0.7200154969274066, 'color': {'r': np.float64(0.8800619877096265), 'g': 1, 'b': 0}}, {'position': {'x': 2.2222222222225696, 'y': 4.444444444444429, 'z': 44.20318981364971}, 'uhi_value': 0.09013528713785447, 'intensity': 0.8840637962729941, 'color': {'r': 1, 'g': np.float64(0.46374481490802344), 'b': 0}}, {'position': {'x': 2.2222222222225696, 'y': 5.5555555555557135, 'z': 37.63453321106948}, 'uhi_value': 0.05815516712961308, 'intensity': 0.7526906642213896, 'color': {'r': 1, 'g': np.float64(0.9892373431144414), 'b': 0}}, {'position': {'x': 2.2222222222225696, 'y': 6.666666666666998, 'z': 6.6726246357948975}, 'uhi_value': -0.092585785218624, 'intensity': 0.13345249271589796, 'color': {'r': 0, 'g': np.float64(0.5338099708635918), 'b': 1}}, {'position': {'x': 2.2222222222225696, 'y': 7.7777777777775725, 'z': 30.079285080918712}, 'uhi_value': 0.02137173384747846, 'intensity': 0.6015857016183742, 'color': {'r': np.float64(0.40634280647349685), 'g': 1, 'b': 0}}, {'position': {'x': 2.2222222222225696, 'y': 8.888888888888857, 'z': 38.83351146061559}, 'uhi_value': 0.06399250505540663, 'intensity': 0.7766702292123118, 'color': {'r': 1, 'g': np.float64(0.8933190831507529), 'b': 0}}, {'position': {'x': 2.2222222222225696, 'y': 10.000000000000142, 'z': 12.917534538204132}, 'uhi_value': -0.062181856348106836, 'intensity': 0.2583506907640826, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.9665972369436695)}}, {'position': {'x': 3.333333333333144, 'y': 0.0, 'z': 35.98130565840267}, 'uhi_value': 0.05010627391841249, 'intensity': 0.7196261131680534, 'color': {'r': np.float64(0.8785044526722134), 'g': 1, 'b': 0}}, {'position': {'x': 3.333333333333144, 'y': 1.1111111111112848, 'z': 31.49851362389333}, 'uhi_value': 0.02828138095943458, 'intensity': 0.6299702724778666, 'color': {'r': np.float64(0.5198810899114665), 'g': 1, 'b': 0}}, {'position': {'x': 3.333333333333144, 'y': 2.2222222222225696, 'z': 27.371540117029348}, 'uhi_value': 0.008188823834664235, 'intensity': 0.5474308023405869, 'color': {'r': np.float64(0.18972320936234777), 'g': 1, 'b': 0}}, {'position': {'x': 3.333333333333144, 'y': 3.333333333333144, 'z': 39.2175370686634}, 'uhi_value': 0.06586216970293632, 'intensity': 0.784350741373268, 'color': {'r': 1, 'g': np.float64(0.862597034506928), 'b': 0}}, {'position': {'x': 3.333333333333144, 'y': 4.444444444444429, 'z': 37.36225378339565}, 'uhi_value': 0.05682954922939222, 'intensity': 0.747245075667913, 'color': {'r': np.float64(0.9889803026716519), 'g': 1, 'b': 0}}, {'position': {'x': 3.333333333333144, 'y': 5.5555555555557135, 'z': 40.518152984483535}, 'uhi_value': 0.07219434012819777, 'intensity': 0.8103630596896707, 'color': {'r': 1, 'g': np.float64(0.7585477612413172), 'b': 0}}, {'position': {'x': 3.333333333333144, 'y': 6.666666666666998, 'z': 29.851835346822842}, 'uhi_value': 0.020264373509997685, 'intensity': 0.5970367069364568, 'color': {'r': np.float64(0.3881468277458273), 'g': 1, 'b': 0}}, {'position': {'x': 3.333333333333144, 'y': 7.7777777777775725, 'z': 27.430067438060274}, 'uhi_value': 0.008473769579825405, 'intensity': 0.5486013487612055, 'color': {'r': np.float64(0.1944053950448219), 'g': 1, 'b': 0}}, {'position': {'x': 3.333333333333144, 'y': 8.888888888888857, 'z': 37.61323292183235}, 'uhi_value': 0.05805146467609375, 'intensity': 0.752264658436647, 'color': {'r': 1, 'g': np.float64(0.9909413662534119), 'b': 0}}, {'position': {'x': 3.333333333333144, 'y': 10.000000000000142, 'z': 23.293019096889445}, 'uhi_value': -0.011667837819558088, 'intensity': 0.4658603819377889, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.13655847224884443)}}, {'position': {'x': 4.444444444445139, 'y': 0.0, 'z': 4.026431048772676}, 'uhi_value': -0.10546902658715096, 'intensity': 0.08052862097545353, 'color': {'r': 0, 'g': np.float64(0.32211448390181413), 'b': 1}}, {'position': {'x': 4.444444444445139, 'y': 1.1111111111112848, 'z': 33.14206766081361}, 'uhi_value': 0.036283177753027585, 'intensity': 0.6628413532162721, 'color': {'r': np.float64(0.6513654128650885), 'g': 1, 'b': 0}}, {'position': {'x': 4.444444444445139, 'y': 2.2222222222225696, 'z': 16.969523152021523}, 'uhi_value': -0.042454370199568256, 'intensity': 0.33939046304043047, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.6424381478382781)}}, {'position': {'x': 4.444444444445139, 'y': 3.333333333333144, 'z': 36.01112711193251}, 'uhi_value': 0.050251462458536525, 'intensity': 0.7202225422386502, 'color': {'r': np.float64(0.8808901689546009), 'g': 1, 'b': 0}}, {'position': {'x': 4.444444444445139, 'y': 4.444444444444429, 'z': 27.58998995874687}, 'uhi_value': 0.00925236735300915, 'intensity': 0.5517997991749374, 'color': {'r': np.float64(0.2071991966997495), 'g': 1, 'b': 0}}, {'position': {'x': 4.444444444445139, 'y': 5.5555555555557135, 'z': 31.668642874355363}, 'uhi_value': 0.02910967115267602, 'intensity': 0.6333728574871073, 'color': {'r': np.float64(0.533491429948429), 'g': 1, 'b': 0}}, {'position': {'x': 4.444444444445139, 'y': 6.666666666666998, 'z': 24.43078637786119}, 'uhi_value': -0.006128511228389643, 'intensity': 0.48861572755722377, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.04553708977110493)}}, {'position': {'x': 4.444444444445139, 'y': 7.7777777777775725, 'z': 24.140010392658603}, 'uhi_value': -0.007544181351659676, 'intensity': 0.48280020785317207, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.06879916858731172)}}, {'position': {'x': 4.444444444445139, 'y': 8.888888888888857, 'z': 40.616176543983364}, 'uhi_value': 0.07267157701027319, 'intensity': 0.8123235308796674, 'color': {'r': 1, 'g': np.float64(0.7507058764813306), 'b': 0}}, {'position': {'x': 4.444444444445139, 'y': 10.000000000000142, 'z': 25.88152328468268}, 'uhi_value': 0.0009345373045953126, 'intensity': 0.5176304656936536, 'color': {'r': np.float64(0.07052186277461425), 'g': 1, 'b': 0}}, {'position': {'x': 5.5555555555557135, 'y': 0.0, 'z': 40.161660077685774}, 'uhi_value': 0.07045872101591215, 'intensity': 0.8032332015537155, 'color': {'r': 1, 'g': np.float64(0.7870671937851381), 'b': 0}}, {'position': {'x': 5.5555555555557135, 'y': 1.1111111111112848, 'z': 40.134362970872075}, 'uhi_value': 0.07032582249425144, 'intensity': 0.8026872594174416, 'color': {'r': 1, 'g': np.float64(0.7892509623302337), 'b': 0}}, {'position': {'x': 5.5555555555557135, 'y': 2.2222222222225696, 'z': 37.95672421833712}, 'uhi_value': 0.05972378423070314, 'intensity': 0.7591344843667425, 'color': {'r': 1, 'g': np.float64(0.9634620625330301), 'b': 0}}, {'position': {'x': 5.5555555555557135, 'y': 3.333333333333144, 'z': 36.53169026021329}, 'uhi_value': 0.05278587291125343, 'intensity': 0.7306338052042658, 'color': {'r': np.float64(0.9225352208170632), 'g': 1, 'b': 0}}, {'position': {'x': 5.5555555555557135, 'y': 4.444444444444429, 'z': 22.156112336026986}, 'uhi_value': -0.01720297488796267, 'intensity': 0.44312224672053974, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.22751101311784105)}}, {'position': {'x': 5.5555555555557135, 'y': 5.5555555555557135, 'z': 44.99746281772253}, 'uhi_value': 0.09400227966408539, 'intensity': 0.8999492563544506, 'color': {'r': 1, 'g': np.float64(0.4002029745821978), 'b': 0}}, {'position': {'x': 5.5555555555557135, 'y': 6.666666666666998, 'z': 0.0}, 'uhi_value': -0.12507208333608788, 'intensity': 0.0, 'color': {'r': 0, 'g': 0, 'b': 1}}, {'position': {'x': 5.5555555555557135, 'y': 7.7777777777775725, 'z': 16.570156623441047}, 'uhi_value': -0.044398723555817914, 'intensity': 0.33140313246882097, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.6743874701247161)}}, {'position': {'x': 5.5555555555557135, 'y': 8.888888888888857, 'z': 23.893257760522914}, 'uhi_value': -0.00874551965650408, 'intensity': 0.4778651552104583, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.08853937915816679)}}, {'position': {'x': 5.5555555555557135, 'y': 10.000000000000142, 'z': 28.592312290288735}, 'uhi_value': 0.014132267519666942, 'intensity': 0.5718462458057747, 'color': {'r': np.float64(0.2873849832230988), 'g': 1, 'b': 0}}, {'position': {'x': 6.666666666666288, 'y': 0.0, 'z': 29.406853597058248}, 'uhi_value': 0.018097938176270707, 'intensity': 0.5881370719411649, 'color': {'r': np.float64(0.35254828776465974), 'g': 1, 'b': 0}}, {'position': {'x': 6.666666666666288, 'y': 1.1111111111112848, 'z': 15.229765607469082}, 'uhi_value': -0.05092454278222017, 'intensity': 0.3045953121493816, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.7816187514024735)}}, {'position': {'x': 6.666666666666288, 'y': 2.2222222222225696, 'z': 34.93166352777088}, 'uhi_value': 0.04499599287706149, 'intensity': 0.6986332705554176, 'color': {'r': np.float64(0.7945330822216703), 'g': 1, 'b': 0}}, {'position': {'x': 6.666666666666288, 'y': 3.333333333333144, 'z': 21.92120681285592}, 'uhi_value': -0.018346634432999415, 'intensity': 0.4384241362571184, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.2463034549715264)}}, {'position': {'x': 6.666666666666288, 'y': 4.444444444444429, 'z': 34.27665382278439}, 'uhi_value': 0.04180701676833898, 'intensity': 0.6855330764556877, 'color': {'r': np.float64(0.7421323058227509), 'g': 1, 'b': 0}}, {'position': {'x': 6.666666666666288, 'y': 5.5555555555557135, 'z': 24.359870807446207}, 'uhi_value': -0.006473770326183421, 'intensity': 0.48719741614892415, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.05121033540430342)}}, {'position': {'x': 6.666666666666288, 'y': 6.666666666666998, 'z': 24.600619255269116}, 'uhi_value': -0.005301663955125727, 'intensity': 0.4920123851053823, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.03195045957847076)}}, {'position': {'x': 6.666666666666288, 'y': 7.7777777777775725, 'z': 47.115781080850006}, 'uhi_value': 0.1043155105679289, 'intensity': 0.9423156216170001, 'color': {'r': 1, 'g': np.float64(0.23073751353199956), 'b': 0}}, {'position': {'x': 6.666666666666288, 'y': 8.888888888888857, 'z': 26.050580048887788}, 'uhi_value': 0.0017576059980265717, 'intensity': 0.5210116009777558, 'color': {'r': np.float64(0.08404640391102314), 'g': 1, 'b': 0}}, {'position': {'x': 6.666666666666288, 'y': 10.000000000000142, 'z': 32.65930642057878}, 'uhi_value': 0.033932809430839324, 'intensity': 0.6531861284115755, 'color': {'r': np.float64(0.6127445136463021), 'g': 1, 'b': 0}}, {'position': {'x': 7.777777777778283, 'y': 0.0, 'z': 24.622795971469852}, 'uhi_value': -0.005193694534832136, 'intensity': 0.49245591942939704, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.03017632228241185)}}, {'position': {'x': 7.777777777778283, 'y': 1.1111111111112848, 'z': 21.162084144275912}, 'uhi_value': -0.02204249425807064, 'intensity': 0.4232416828855182, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.3070332684579271)}}, {'position': {'x': 7.777777777778283, 'y': 2.2222222222225696, 'z': 9.05261566560588}, 'uhi_value': -0.08099857593720175, 'intensity': 0.18105231331211757, 'color': {'r': 0, 'g': np.float64(0.7242092532484703), 'b': 1}}, {'position': {'x': 7.777777777778283, 'y': 3.333333333333144, 'z': 35.357830852902126}, 'uhi_value': 0.04707082842196347, 'intensity': 0.7071566170580424, 'color': {'r': np.float64(0.8286264682321698), 'g': 1, 'b': 0}}, {'position': {'x': 7.777777777778283, 'y': 4.444444444444429, 'z': 34.839291128027945}, 'uhi_value': 0.04454626919565183, 'intensity': 0.6967858225605589, 'color': {'r': np.float64(0.7871432902422355), 'g': 1, 'b': 0}}, {'position': {'x': 7.777777777778283, 'y': 5.5555555555557135, 'z': 39.69668145580141}, 'uhi_value': 0.0681949290370032, 'intensity': 0.7939336291160283, 'color': {'r': 1, 'g': np.float64(0.8242654835358869), 'b': 0}}, {'position': {'x': 7.777777777778283, 'y': 6.666666666666998, 'z': 29.636416141099403}, 'uhi_value': 0.019215584926893875, 'intensity': 0.5927283228219881, 'color': {'r': np.float64(0.37091329128795225), 'g': 1, 'b': 0}}, {'position': {'x': 7.777777777778283, 'y': 7.7777777777775725, 'z': 17.53443825892423}, 'uhi_value': -0.03970402808149794, 'intensity': 0.3506887651784846, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.5972449392860617)}}, {'position': {'x': 7.777777777778283, 'y': 8.888888888888857, 'z': 18.67609943084109}, 'uhi_value': -0.03414574371749129, 'intensity': 0.37352198861682184, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.5059120455327126)}}, {'position': {'x': 7.777777777778283, 'y': 10.000000000000142, 'z': 23.076256559015004}, 'uhi_value': -0.012723166541079706, 'intensity': 0.4615251311803001, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.15389947527879966)}}, {'position': {'x': 8.888888888888857, 'y': 0.0, 'z': 35.889756776770845}, 'uhi_value': 0.04966055961207232, 'intensity': 0.717795135535417, 'color': {'r': np.float64(0.8711805421416678), 'g': 1, 'b': 0}}, {'position': {'x': 8.888888888888857, 'y': 1.1111111111112848, 'z': 47.02554373649215}, 'uhi_value': 0.10387618160366921, 'intensity': 0.9405108747298431, 'color': {'r': 1, 'g': np.float64(0.2379565010806277), 'b': 0}}, {'position': {'x': 8.888888888888857, 'y': 2.2222222222225696, 'z': 17.722345782464537}, 'uhi_value': -0.03878918270034193, 'intensity': 0.35444691564929076, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.582212337402837)}}, {'position': {'x': 8.888888888888857, 'y': 3.333333333333144, 'z': 17.898356098702138}, 'uhi_value': -0.03793225998744568, 'intensity': 0.35796712197404273, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.5681315121038291)}}, {'position': {'x': 8.888888888888857, 'y': 4.444444444444429, 'z': 31.25357881779895}, 'uhi_value': 0.02708889291096255, 'intensity': 0.625071576355979, 'color': {'r': np.float64(0.500286305423916), 'g': 1, 'b': 0}}, {'position': {'x': 8.888888888888857, 'y': 5.5555555555557135, 'z': 20.691015745360758}, 'uhi_value': -0.024335934886932115, 'intensity': 0.41382031490721516, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.34471874037113936)}}, {'position': {'x': 8.888888888888857, 'y': 6.666666666666998, 'z': 20.088862426978565}, 'uhi_value': -0.027267574725991298, 'intensity': 0.4017772485395713, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.39289100584171477)}}, {'position': {'x': 8.888888888888857, 'y': 7.7777777777775725, 'z': 33.53863588627178}, 'uhi_value': 0.03821390730838685, 'intensity': 0.6707727177254356, 'color': {'r': np.float64(0.6830908709017423), 'g': 1, 'b': 0}}, {'position': {'x': 8.888888888888857, 'y': 8.888888888888857, 'z': 48.84369382725632}, 'uhi_value': 0.11272801564120943, 'intensity': 0.9768738765451264, 'color': {'r': 1, 'g': np.float64(0.09250449381949455), 'b': 0}}, {'position': {'x': 8.888888888888857, 'y': 10.000000000000142, 'z': 35.789172882145174}, 'uhi_value': 0.04917085749858966, 'intensity': 0.7157834576429034, 'color': {'r': np.float64(0.8631338305716136), 'g': 1, 'b': 0}}, {'position': {'x': 9.999999999999432, 'y': 0.0, 'z': 41.69645300458059}, 'uhi_value': 0.07793100415682354, 'intensity': 0.8339290600916118, 'color': {'r': 1, 'g': np.float64(0.6642837596335527), 'b': 0}}, {'position': {'x': 9.999999999999432, 'y': 1.1111111111112848, 'z': 30.895252015330634}, 'uhi_value': 0.025344345305809188, 'intensity': 0.6179050403066126, 'color': {'r': np.float64(0.47162016122645056), 'g': 1, 'b': 0}}, {'position': {'x': 9.999999999999432, 'y': 2.2222222222225696, 'z': 35.54911631583699}, 'uhi_value': 0.04800211961750049, 'intensity': 0.7109823263167399, 'color': {'r': np.float64(0.8439293052669594), 'g': 1, 'b': 0}}, {'position': {'x': 9.999999999999432, 'y': 3.333333333333144, 'z': 37.11536076065887}, 'uhi_value': 0.05562752741975485, 'intensity': 0.7423072152131773, 'color': {'r': np.float64(0.9692288608527093), 'g': 1, 'b': 0}}, {'position': {'x': 9.999999999999432, 'y': 4.444444444444429, 'z': 40.234361952509964}, 'uhi_value': 0.07081267690405958, 'intensity': 0.8046872390501992, 'color': {'r': 1, 'g': np.float64(0.7812510437992031), 'b': 0}}, {'position': {'x': 9.999999999999432, 'y': 5.5555555555557135, 'z': 33.05385187228513}, 'uhi_value': 0.03585369092269849, 'intensity': 0.6610770374457026, 'color': {'r': np.float64(0.6443081497828103), 'g': 1, 'b': 0}}, {'position': {'x': 9.999999999999432, 'y': 6.666666666666998, 'z': 20.232837313362403}, 'uhi_value': -0.0265666195043529, 'intensity': 0.40465674626724807, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.3813730149310077)}}, {'position': {'x': 9.999999999999432, 'y': 7.7777777777775725, 'z': 22.811493310519037}, 'uhi_value': -0.01401219121887255, 'intensity': 0.45622986621038075, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.175080535158477)}}, {'position': {'x': 9.999999999999432, 'y': 8.888888888888857, 'z': 9.945154492679134}, 'uhi_value': -0.07665316704634795, 'intensity': 0.19890308985358265, 'color': {'r': 0, 'g': np.float64(0.7956123594143306), 'b': 1}}, {'position': {'x': 9.999999999999432, 'y': 10.000000000000142, 'z': 16.196579009784966}, 'uhi_value': -0.04621752116390433, 'intensity': 0.32393158019569934, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.7042736792172026)}}]

# Scene settings
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 128

# Create terrain plane
bpy.ops.mesh.primitive_plane_add(size=200, location=(50, 50, 0))
terrain = bpy.context.active_object
terrain.name = "UHI_Terrain"

# Create terrain material
terrain_mat = bpy.data.materials.new(name="TerrainMaterial")
terrain_mat.use_nodes = True
terrain_bsdf = terrain_mat.node_tree.nodes["Principled BSDF"]
terrain_bsdf.inputs["Base Color"].default_value = (0.15, 0.15, 0.18, 1)
terrain_bsdf.inputs["Roughness"].default_value = 0.8
terrain.data.materials.append(terrain_mat)

# Subdivide terrain for detail
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.subdivide(number_cuts=20)
bpy.ops.object.mode_set(mode='OBJECT')

# Create hotspot pillar material
def create_hotspot_material(name, color, emission_strength):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    for node in nodes:
        nodes.remove(node)
    
    # Add nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    emission = nodes.new('ShaderNodeEmission')
    mix = nodes.new('ShaderNodeMixShader')
    
    # Set node positions
    output.location = (400, 0)
    mix.location = (200, 0)
    principled.location = (0, 100)
    emission.location = (0, -100)
    
    # Configure nodes
    principled.inputs["Base Color"].default_value = (*color, 1)
    principled.inputs["Roughness"].default_value = 0.4
    principled.inputs["Metallic"].default_value = 0.6
    
    emission.inputs["Color"].default_value = (*color, 1)
    emission.inputs["Strength"].default_value = emission_strength
    
    mix.inputs["Fac"].default_value = 0.3
    
    # Link nodes
    links.new(principled.outputs["BSDF"], mix.inputs[1])
    links.new(emission.outputs["Emission"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])
    
    return mat

# Create hotspot pillars
pillar_collection = bpy.data.collections.new("UHI_Hotspots")
bpy.context.scene.collection.children.link(pillar_collection)

for i, point in enumerate(hotspot_data):
    pos = point['position']
    color = (point['color']['r'], point['color']['g'], point['color']['b'])
    height = max(point['position']['z'], 1)
    intensity = point['intensity']
    
    # Create cylinder
    bpy.ops.mesh.primitive_cylinder_add(
        radius=1.5,
        depth=height,
        location=(pos['x'], pos['y'], height/2)
    )
    pillar = bpy.context.active_object
    pillar.name = f"Hotspot_{i:04d}"
    
    # Apply material
    mat_name = f"HotspotMat_{i:04d}"
    emission = 2.0 if intensity > 0.7 else 0.5
    mat = create_hotspot_material(mat_name, color, emission)
    pillar.data.materials.append(mat)
    
    # Move to collection
    bpy.context.scene.collection.objects.unlink(pillar)
    pillar_collection.objects.link(pillar)
    
    # Add glow ring for high intensity
    if intensity > 0.7:
        bpy.ops.mesh.primitive_torus_add(
            major_radius=2.5,
            minor_radius=0.2,
            location=(pos['x'], pos['y'], 0.2)
        )
        glow = bpy.context.active_object
        glow.name = f"Glow_{i:04d}"
        
        glow_mat = bpy.data.materials.new(name=f"GlowMat_{i:04d}")
        glow_mat.use_nodes = True
        glow_bsdf = glow_mat.node_tree.nodes["Principled BSDF"]
        glow_bsdf.inputs["Emission"].default_value = (*color, 1)
        glow_bsdf.inputs["Emission Strength"].default_value = 5
        glow.data.materials.append(glow_mat)
        
        bpy.context.scene.collection.objects.unlink(glow)
        pillar_collection.objects.link(glow)

# Add lighting
# Sun light
bpy.ops.object.light_add(type='SUN', location=(100, 100, 100))
sun = bpy.context.active_object
sun.name = "UHI_Sun"
sun.data.energy = 3

# Area light for fill
bpy.ops.object.light_add(type='AREA', location=(0, 0, 80))
area = bpy.context.active_object
area.name = "UHI_AreaLight"
area.data.energy = 500
area.data.size = 50

# Add camera
bpy.ops.object.camera_add(location=(150, -100, 100))
camera = bpy.context.active_object
camera.name = "UHI_Camera"
camera.rotation_euler = (math.radians(60), 0, math.radians(45))
bpy.context.scene.camera = camera

# Add HDRI world (optional - uses solid color if no HDRI)
world = bpy.data.worlds.new(name="UHI_World")
bpy.context.scene.world = world
world.use_nodes = True
world_nodes = world.node_tree.nodes
bg_node = world_nodes["Background"]
bg_node.inputs["Color"].default_value = (0.05, 0.05, 0.1, 1)
bg_node.inputs["Strength"].default_value = 0.5

# Set render settings
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.film_transparent = False

print(f"Created {len(hotspot_data)} hotspot pillars")
print("Scene setup complete! Press F12 to render.")
