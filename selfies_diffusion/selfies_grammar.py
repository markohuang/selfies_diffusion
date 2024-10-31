from group_selfies import (
    Group, 
    GroupGrammar, 
)

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 

g0 = Group('carboxylate', '*1C(=O)([O-])')
g1 = Group('cyclopentane', 'C1CCCC1', all_attachment=True)
g2 = Group('tetrahydrofuran', 'C1CCOC1', all_attachment=True)
g3 = Group('pyrrolidine', 'C1CCNC1', all_attachment=True)
g4 = Group('dithiole', 'C(*2)1SC(*1)=C(*1)S1')
g5 = Group('imidazoline', 'C1N(*1)C=CN(*1)1')
g6 = Group('cyclobutane', 'C(*1)1C(*1)C(*1)C(*1)1')
g7 = Group('cyclopropane', 'C(*1)1C(*1)C(*1)1')
g14 = Group('phosphineoxide', 'P=O', all_attachment=True)
g15 = Group('triazinane', 'C(*1)1NC(*1)NC(*1)N1')
g16 = Group('diazinane', 'C1N(*1)CCN(*1)C1')
g17 = Group('piperidine', 'C1N(*1)CCC(*1)C1')
g18 = Group('cyclohexane', 'C(*1)1C(*1)C(*1)C(*1)C(*1)C(*1)1', all_attachment=True)
g19 = Group('tetrazine', 'N1=NC(*1)=NN=C(*1)1')
g21 = Group('pyrimidine', 'C1=NC(*1)=CC(*1)=N1')
g22 = Group('pyrazine', 'N1=C(*1)C(*1)=NC(*1)=C(*1)1')
g24 = Group('benzene', 'C1=CC=CC=C1', all_attachment=True)
g25 = Group('cyclopentadiene', 'C1C(*1)=CC=C(*1)1')
g26 = Group('furan', 'O1C(*1)=CC=C(*1)1')
g27 = Group('thiophene', 'S1C(*1)=CC=C(*1)1')
g28 = Group('pyrrole', 'N1C(*1)=CC=C(*1)1')
g29 = Group('pyrazole', 'N1=CC=CN1', all_attachment=True)
g32 = Group('triazole1', 'N1=NNC(*1)=C(*1)1')
g33 = Group('tetrazole', 'N(*1)1N=NC(*1)=N1')
g34 = Group('oxadiazole', 'O1C(*1)=NN=C(*1)1')
group_list = [g0, g1, g2, g3, g4, g5, g6, g7, g14, g15, g16, g17, g18, g19, g21, g22, g24, g25, g26, g27, g28, g29, g32, g33, g34]
grammar = GroupGrammar(group_list) | GroupGrammar.essential_set()

__all__ = ['grammar']