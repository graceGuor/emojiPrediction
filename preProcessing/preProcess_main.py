import sys
import preProcessing.filterSpecCha as filterSpecCha
import preProcessing.lowerCha as lowerCha
import preProcessing.removeStopwords as removeStopwords
import preProcessing.replaceSparseWithUnk as replaceSparseWithUnk
import preProcessing.delUnkProportion as delUnkProportion
import preProcessing.dataDivision as dataDivision

# filterSpecCha.filterSpecCha_main()
lowerCha.lowerCha_main()
removeStopwords.removeStopwords_main()
replaceSparseWithUnk.replaceSparseWithUnk_main()
delUnkProportion.delUnkProportion_main()
dataDivision.dataDivision811_main()
print(sys.argv[0] + 'Finished!')