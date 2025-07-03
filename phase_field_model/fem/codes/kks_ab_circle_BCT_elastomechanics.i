###########################################################################################################################
# The multicomponent is represented by the suffixes a and b for ternary system A-B-C
# For multiphase system with eta1, eta2, and eta3, the suffixes 1,2 and 3 come after the variables and materials properties  
############################################################################################################################
[Mesh]
  type = GeneratedMesh
  dim = 2
  nx = 100 #15 #20 #25 #15 #35 #30 #25 #16 #20 #15 #20 #100 #30 #20 #25 #20 #40 #50 #20
  ny = 100 #15 #20 #25 #15 #35 #30 #25 #16 #20 #15 #20 #100 #30 #20 #25 #20 #40 #50 #20
  nz = 0
  xmin = 0
  xmax = 500 #80
  ymin = 0
  ymax = 500 #50
  zmin = 0
  zmax = 0
  elem_type = QUAD4
[]
#########################################################################################


#########################################################################################
[BCs]
  [./Periodic]
    [./all]
      #auto_direction = 'x y'
       auto_direction = 'y'
     [../]
   [../]
 #################################
 #Phase field
 #####################################
  [./neumann1]
        type = NeumannBC
        boundary = 'right'
        variable = 'eta1'
        value = 0
    [../]
    [./neumann2]
       type = NeumannBC
        boundary = 'left'
        variable = 'eta2'
        value = 0
   [../]
################################################################################
# electric potential
###############################################################################
  #[./neumann_pota] #anode Cu
  #     type = NeumannBC
   #     boundary = 'left'
  #      variable = 'pot'
   #     value = 8.50E-11 #-17.0E-11 #-17.0E-11 #j=5.0E+2 A/cm2=5.0E+6 A/m2 = 5.0E-12 A/nm2 grad_u = -j*rho_cu V/m i.e. factor= 1.0/(length_scale)
   #[../]
  # [./dir_potmid] #anode Sn
    #   type = DirichletBC
    #    boundary = 'right'
    #    variable = 'pot'
    #    value = 0.005 #5.0E+2 A/cm2=5.0E+6 A/m2 = 5.0E-12 A/nm2 i.e. factor= 1.0/(length_scale^2)
   #[../]
   [./dir_potright] #anode Cu
       type = DirichletBC
        boundary = 'right'
        variable = 'pot'
        value = 0.005 #0.005 #5.0E+2 A/cm2=5.0E+6 A/m2 = 5.0E-12 A/nm2 i.e. factor= 1.0/(length_scale^2)
   [../]
   #[./neumann_potmid] #anode Sn
   #    type = NeumannBC
   #     boundary = 'right'
   #     variable = 'pot'
   #     value = -5.50E-10 #-1.75E-8 #-11.0E-10 #-11.0E-10 #5.0E+2 A/cm2=5.0E+6 A/m2 = 5.0E-12 A/nm2 i.e. factor= 1.0/(length_scale^2)
   #[../]
   #[./neumann_potmid] #anode Sn
    #   type = NeumannBC
    #    boundary = 'right'
    #    variable = 'pot'
    #    value = -11.0E-10 #-11.0E-10 #5.0E+2 A/cm2=5.0E+6 A/m2 = 5.0E-12 A/nm2 i.e. factor= 1.0/(length_scale^2)
   #[../]
  
   [./dir_potleft] #anode Cu
       type = DirichletBC
        boundary = 'left'
        variable = 'pot'
        value = 0.001 #j=5.0E+2 A/cm2=5.0E+6 A/m2 = 5.0E-12 A/nm2 grad_u = -j*rho_cu V/m i.e. factor= 1.0/(length_scale)
   [../]
   #[./dir_potmid] #anode Sn
   #    type = DirichletBC
   #     boundary = 'right'
   #     variable = 'pot'
   #     value = 0 #5.0E+2 A/cm2=5.0E+6 A/m2 = 5.0E-12 A/nm2 i.e. factor= 1.0/(length_scale^2)
   #[../]
 ######################################################################################
 # displacement (for stress free strain, displacement boundary conditions are zero)
 ######################################################################################
     [./right_x]
      type = DirichletBC
      variable = disp_x
      boundary = right
      value = 0
    [../]

      [./left_x]
        type = DirichletBC
        variable = disp_x
        boundary = left
        value = 0
      [../]     

      [./top_x]
        type = DirichletBC
        variable = disp_x
        boundary = top
        value = 0
      [../]
  
        [./bottom_x]
          type = DirichletBC
          variable = disp_x
          boundary = bottom
          value = 0
        [../]   
###############################
      [./right_y]
        type = DirichletBC
        variable = disp_y
        boundary = right
        value = 0
      [../]
  
        [./left_y]
          type = DirichletBC
          variable = disp_y
          boundary = left
          value = 0
        [../]     
    
      [./top_y]
        type = DirichletBC
        variable = disp_y
        boundary = top
        value = 0
      [../]
  
        [./bottom_y]
          type = DirichletBC
          variable = disp_y
          boundary = bottom
          value = 0
        [../]  
[]
#########################################################################################


#########################################################################################
[AuxVariables]
  [./bnds]
  [../]
  [./Energy]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./gr_c]
   order = CONSTANT
   family = MONOMIAL
  [../]
  ##################
# # Stress due to elasticity considerations
[./von_mises]
  #Dependent variable used to visualize the Von Mises stress
  order = CONSTANT
  family = MONOMIAL
[../]
[./sigma11]
  order = CONSTANT
  family = MONOMIAL
[../]
[./sigma22]
  order = CONSTANT
  family = MONOMIAL
[../]
[./sigma12]
  order = CONSTANT
  family = MONOMIAL
[../]
[./e11]
  order = CONSTANT
  family = MONOMIAL
[../]
[./e12]
  order = CONSTANT
  family = MONOMIAL
[../]
[./e22]
  order = CONSTANT
  family = MONOMIAL
[../]
[./e33]
  order = CONSTANT
  family = MONOMIAL
[../]

[]
#########################################################################################


#########################################################################################

[Variables]
# For A-B-C alloy, cA+cB+cC = 1
# So, DOF = 2 i.e. cA and cB are sufficient to describe the composition of the ternary system A-B-C
############################################
  # potential variable used in SplitCHCRes and kkssplitchcres (global)
  [./wa] #A 
    order = FIRST
    family = LAGRANGE
    #caling = 1.0E-02
  [../]
  
#  [./wb] # B
#    order = FIRST
#    family = LAGRANGE
#    scaling = 1.0E-02
#  [../]
############################################
 
############################################
  # concentration (global) of A
  [./ca]
    order = FIRST
    family = LAGRANGE
    scaling = 1.0E-04
  [../]
  
   # concentration (global) of B
 # [./cb]
 #   order = FIRST
 #   family = LAGRANGE
 #   ###scaling = 1.0E+02
 # [../]
############################################

################################################################ 
# Phase concentrations corresponding to global composition cA
############################################################### 
  # phase concentration 1
  [./c1a]
    order = FIRST
    family = LAGRANGE
    #initial_condition = 0.2
    #scaling = 1.0E+04
  [../]
######################

######################
  # phase concentration 2
  [./c2a]
    order = FIRST
    family = LAGRANGE
    #initial_condition = 0.5
    #scaling = 1.0E+04
  [../]
######################

######################
  # phase concentration 3
  #[./c3a]
  #  order = FIRST
  #  family = LAGRANGE
  #  #initial_condition = 0.8
  #  #scaling = 1.0E+04
  #[../]
############################################
################################################################ 
# Phase concentrations corresponding to global composition cB
############################################################### 
  # phase concentration 1
 # [./c1b]
 #   order = FIRST
 #   family = LAGRANGE
 #   #initial_condition = 0.2
 #   ###scaling = 1.0E+02
 # [../]
######################

######################
  # phase concentration 2
#  [./c2b]
#    order = FIRST
#    family = LAGRANGE
#    #initial_condition = 0.5
#    ###scaling = 1.0E+02
#  [../]
######################

######################
  # phase concentration 3
 # [./c3b]
 #   order = FIRST
 #   family = LAGRANGE
 #   #initial_condition = 0.8
 #   ###scaling = 1.0E+02
 # [../]
############################################


############################################
  # order parameter 1 LI2SN5
  [./eta1]
    order = FIRST
    family = LAGRANGE
    scaling = 1.0E-04
  [../]
######################

######################
  # order parameter 2 BCT SN
  [./eta2]
    order = FIRST
    family = LAGRANGE
    scaling = 1.0E-04
  [../]
######################


 ###################
     [./pot] 
        initial_condition = 0.000 # V unit 1 mV
        order = FIRST
        family = LAGRANGE
        scaling=1.0E+0
    [../]
############################

#### For Elasticity
# Displacement variables for NS equation 
 [./disp_x]
  scaling=1.0E-05 
  [../]
  [./disp_y]
    scaling=1.0E-05 
  [../]
[]
#########################################################################################
###########Upto here#####################################


#########################################################################################

[ICs]

############################################
    [./eta1]  # extends from x = 250 to y = 500 , x no change
        variable = eta1
        type = FunctionIC
        function = 'r:=sqrt((x-500)^2+(y-250)^2);if(r<=50,1,0)'
        #function = 'if(x>=250,1,0)'
        #function = 'if(y<=250,1,0)'
    [../]
######################

 [./eta2] # extends from y = 20 to y = 30 , x no change
        variable = eta2
        type = FunctionIC
        function = 'r:=sqrt((x-500)^2+(y-250)^2);if(r<=50,0,1)'
        #function = 'if(x>=250,0,1)'
        #function = 'if(y>30,1,0)'
        #function = 'if(y<=200,0,if(y>250&y<=250&x>15&x<=75,0,if(y>200&y<=250&x>105&x<=195,0,if(y>200&y<=250&x>225&x<=315,0,if(y>200&y<=250&x>345&x<=385,0,if(y>200&y<=250&x>415&x<=485,0,1))))))'
       #function = 'r2:=sqrt((x-60)^2+(y-250)^2); r4:=sqrt((x-165)^2+(y-250)^2); r5:=sqrt((x-270)^2+(y-250)^2); r6:=sqrt((x-370)^2+(y-250)^2); r7:=sqrt((x-460)^2+(y-250)^2); r8:=sqrt((x-555)^2+(y-250)^2);if(y<=250,0,if(r2<=45,0,if(r4<=30,0,if(r5<=45,0,if(r6<=25,0,if(r7<=35,0,1))))))'
         #function = 'if(y<250,0,if(y>=250&y<=340&x>15&x<=105, 0, if(y>=340&y<=385.0&x>37.5&x<=82.5,0,if(y>=250&y<=310&x>135&x<=195,0,if(y>=310&y<=340.0&x>150.0&x<=180.0,0,if(y>=250&y<=340&x>225&x<=315,0,if(y>=340&y<=385.0&x>247.5&x<=292.5,0,if(y>=250&y<=300&x>345&x<=395,0,if(y>=300&y<=325.0&x>357.5&x<=382.5,0,if(y>=250&y<=320&x>425&x<=495,0,if(y>=320&y<=355.0&x>442.5&x<=477.5,0,1)))))))))))'
 [../]

############################################
    [./ca] #Global Composition of A for ternary A-B-C alloy
        variable = ca
        type = FunctionIC
        function = 'r:=sqrt((x-500)^2+(y-250)^2); 0.71*if(r<=50,1,0)+0.95*if(r<=50,0,1)'
        #function = '0.71*if(x>=250,1,0)+0.95*if(x>=250,0,1)'
        #function = 'if(y<=20,0.01,0)+if(y>20&y<=30,0.667,0)+if(y>30,0.95,0)'
        #function = 'r2:=sqrt((x-60)^2+(y-250)^2); r4:=sqrt((x-165)^2+(y-250)^2); r5:=sqrt((x-270)^2+(y-250)^2); r6:=sqrt((x-370)^2+(y-250)^2); r7:=sqrt((x-460)^2+(y-250)^2); r8:=sqrt((x-555)^2+(y-250)^2); 0.01*if(y<=250,1,0)+0.455*if(y<=250,0,if(r2<=45,1,if(r4<=30,1,if(r5<=45,1,if(r6<=25,1,if(r7<=35,1,1))))))+0.95*if(y<=250,0,if(r2<=45,0,if(r4<=30,0,if(r5<=45,0,if(r6<=25,0,if(r7<=35,0,1))))))'  
        #function = '0.71*if(y<=250,1,0)+0.71*if(y<250,0,if(y>=250&y<=340&x>15&x<=105, 1, if(y>=340&y<=385.0&x>37.5&x<=82.5,1,if(y>=250&y<=310&x>135&x<=195,1,if(y>=310&y<=340.0&x>150.0&x<=180.0,1,if(y>=250&y<=340&x>225&x<=315,1,if(y>=340&y<=385.0&x>247.5&x<=292.5,1,if(y>=250&y<=300&x>345&x<=395,1,if(y>=300&y<=325.0&x>357.5&x<=382.5,1,if(y>=250&y<=320&x>425&x<=495,1,if(y>=320&y<=355.0&x>442.5&x<=477.5,1,0)))))))))))+0.95*if(y<250,0,if(y>=250&y<=340&x>15&x<=105, 0, if(y>=340&y<=385.0&x>37.5&x<=82.5,0,if(y>=250&y<=310&x>135&x<=195,0,if(y>=310&y<=340.0&x>150.0&x<=180.0,0,if(y>=250&y<=340&x>225&x<=315,0,if(y>=340&y<=385.0&x>247.5&x<=292.5,0,if(y>=250&y<=300&x>345&x<=395,0,if(y>=300&y<=325.0&x>357.5&x<=382.5,0,if(y>=250&y<=320&x>425&x<=495,0,if(y>=320&y<=355.0&x>442.5&x<=477.5,0,1)))))))))))'    
 [../]
############################################  
 #[./cb] #Global Composition of B for ternary A-B-C alloy
 #       variable = cb
 #       type = FunctionIC
 #       #function = 'if(y<=20,0.01,0)+if(y>20&y<=30,0.667,0)+if(y>30,0.95,0)'
 #       function = '0.94*if(y<=20,1,0)+0.267*if(y>20&y<=30&x>5&x<=15,1,if(y>20&y<=35&x>20&x<=30,1,if(y>20&y<=30&x>35&x<=45,1,0)))+0.02*if(y<=20,0,if(y>20&y<=30&x>5&x<=15,0,if(y>20&y<=35&x>20&x<=30,0,if(y>20&y<=30&x>35&x<=45,0,1))))'      
# [../]

[]

########################################################################################
#####upto here####################################################################### 08.05.2023 Monday

########################################################################################

[Materials]
##When kappa is increased by a factor of f_kappa=energy_scale/length_scale, simulation converges at factor_f1 = 1.0E+07
# When f_kappa=1, factor_f1 = 1.0E+09
############################################
##Chemical and Electromigration Free Energy
##############################################
  # Free energy of LI2SN5 phase 
  [./f1] # this phase is expected to shrink
    type = DerivativeParsedMaterial
    f_name = f1chemem #F1
    args = 'c1a pot'
    material_property_names = 'length_scale energy_scale Nav echarge zeff1'
    constant_names = 'factor_f1 factor_e1'
    constant_expressions = '1.00E+05 1.0e+02' #'1.0E+08' '1.0E+08'
    function = '(energy_scale/(length_scale)^3) *(1.239e+04*(c1a-0.71)^2*(c1a-0.95)^2-1.46e+03)*factor_f1+(energy_scale/length_scale^3)*(Nav*echarge*pot*(zeff1*(c1a-0.71)))*factor_e1'
    #constant_expressions = '1.00E+04' #'1.0E+08' '1.0E+08'
    #function = '(energy_scale/(length_scale)^3) *(70.6*(c1a-0.2)^2 + 7.2*(c1a-0.2)-9.7)*factor_f1'
    #function = '1.0E+05*(-0.01+0.02*(c1a-0.2)^2)'
  [../]
######################

######################
  [./f2] # this phase is expected to grow
    type = DerivativeParsedMaterial
    f_name = f2chemem #F2
    args = 'c2a pot'
    material_property_names = 'length_scale energy_scale Nav echarge zeff2'
    constant_names = 'factor_f2 factor_e2'
    constant_expressions = '1.00E+05 1.00e+02' #'1.0E+08' '1.0E+08'
    function = '(energy_scale/(length_scale)^3)*(1.239e+04*(c2a-0.71)^2*(c2a-0.95)^2-1.46e+03)*factor_f2 + (energy_scale/length_scale^3)*(Nav*echarge*pot*(zeff2*(c2a-0.95)))*factor_e2'
    #function = '(energy_scale/(length_scale)^3) *(2.55e+04*(c2a-0.39)^2+1.65e+03*(c2a-0.39)-2.37e+03-22.5e+03)*factor_f2' #The term -9.50e+03 accounts for compensation of dc/dr
    #function = '(energy_scale/(length_scale)^3) *(2.347e+04*(c2a-0.447)^2+1.65e+03*(c2a-0.447)-1.92e+03-8.50e+03)*factor_f2' #The term -8.50e+03 accounts for compensation of dc/dr
    #function = '(energy_scale/(length_scale)^3) *(2.347e+04*(c2a-0.38)^2+1.65e+03*(c2a-0.38)-1.92e+03-8.50e+03)*factor_f2' #The term -8.50e+03 accounts for compensation of dc/dr
    #constant_expressions = '1.0E+04' #'1.0E+08' '1.0E+08'
    #function = '(energy_scale/(length_scale)^3) *(690.00*(c2a-0.667)^2+290.00*(c2a-0.667)-20.0)*factor_f2'
    #function = '(energy_scale/(length_scale)^3) *(95.4*(c2a-0.667)^2+34*(c2a-0.667)-20.0)*factor_f2'
  [../]
######################

######################
#  [./f3] # this phase is expected to shrink
#    type = DerivativeParsedMaterial
#    f_name = F3
#    args = 'c3a'
#    material_property_names = 'length_scale energy_scale'
#    constant_names = 'factor_f3'
#    constant_expressions = '1.00E+06' #'1.0E+08' '1.0E+08'
#    #function = '(energy_scale/(length_scale)^3) *(75.0*(c3a-0.95)^2+15*(c3a-0.95)-8.8)*factor_f3'
#    function = '(energy_scale/(length_scale)^3) *(1.921e+04*(c3a-0.90)^2+0.541e+03*(c3a-0.90)-1.66e+03)*factor_f3'
#    #constant_expressions = '1.00E+04' #'1.0E+08' '1.0E+08'
#    #function = '(energy_scale/(length_scale)^3) *(75.0*(c3a-0.95)^2+15*(c3a-0.95)-8.8)*factor_f3'
#  [../]
############################################
# Elastic Free Energy
#############################################
# The f_el_scaled = 6.24*f_el (with C_ijkl in GPa)
# es/ls^3=6.24*10^18/10^27=6.24E-09
# 1GPa = 10^9 Pa
#f_el_scale=6.24E-09*1.0E+09=6.24*f_el(C_ijkl in GPa)
############################################    
# Elastic properties to be used in NS
# C_ijkl = '1111 1122 1133 2222 2233 3333 2323 3131 1212' for the symmetric9
# C_ijkl = '11 12 13 22 23 33 44 55 66' for the symmetric9
# Reference: https://mooseframework.inl.gov/source/materials/ComputeElasticityTensor.html
############################################
  [./elasticity_tensor_precipitate] # LI2SN5 eta1
    type = ComputeElasticityTensor
    base_name = C_eta1
    fill_method = symmetric9
    # C_ijkl = '1201.92 493.44 455.52 1258.88 417.28 1172.32 493.44 411.84 398.4'  #'193e9 79e9 73e9 202e9 67e9 188e9 79e9 66e9 64e9'  #'1.1e6 1e5 0 1e6 0 1e6 .5e6 .2e6 .5e6' 
    C_ijkl = '60.08 55.75 19.55 60.08 19.55 97.48 20.44 20.44 14.93' # Zhang et al, 2016, JEM, 2016, 330:111-119.
    [../]
  [./strain_1]
    type = ComputeSmallStrain
    base_name = C_eta1
    eigenstrain_names = 'C_eigenstrain'
    displacements = 'disp_x disp_y'
  [../]
  [./stress_1]
    type = ComputeLinearElasticStress
    base_name = C_eta1
  [../]
  [./eigenstrain_1]
    type = ComputeEigenstrain
    base_name = C_eta1
    eigen_base = '7.32e-2' # '0.1 0.05 0 0 0 0.01' #Wu and Ji, 2025 , arxiv.org/pdf/2503.17484
    #prefactor = -1
    prefactor = -1.0E-03 # Refer the compensation made for scaling f_el in the same units of fchemem
    eigenstrain_name = 'C_eigenstrain'
  [../]

    ##


    [./fel_eta1]
      type = ElasticEnergyMaterial
      args = ' '
      base_name = C_eta1
      f_name = fel1
      outputs = exodus
      output_properties = fel1
    [../]
###########################################

####################################################################################################################################
# Elastic properties to be used in NS
# C_ijkl = '1111 1122 1133 2222 2233 3333 2323 3131 1212' for the symmetric9
# C_ijkl = '11 12 13 22 23 33 44 55 66' for the symmetric9
# Reference: https://mooseframework.inl.gov/source/materials/ComputeElasticityTensor.html
############################################
 [./elasticity_tensor_matrix]   # BCT Sn 
    type = ComputeElasticityTensor
    base_name = C_eta2
    fill_method = symmetric9
    #C_ijkl =''1111 1122 1133 2222 2233 3333 2323 3131 1212 '
    # C_ijkl = '1152.4 355.68 355.68 1060.8 443.04 1060.8 312 162.24 162.24'  #'1.1e6 1e5 0 1e6 0 1e6 .5e6 .2e6 .5e6'  '1152.4 355.68 355.68 1060.8 443.04 1060.8 312 162.24 162.24'
    # C_ijkl = '832.44 314.00 275.87 832.44 275.87 811.40 170.62 170.62 181.69'  # @ 723 K
    C_ijkl = '7.2E+1 5.85E+1 3.74E+1 7.2E+1 3.74E+1  8.8E+1 2.19E+1 2.19E+1 2.4E+1'  # @ 0 K  Unit: GPa Jiang et al., 2019, JEM 2019, 48:8076-8088
    [../]
  [./strain_2]  # al2cu
    type = ComputeSmallStrain
    base_name = C_eta2
    eigenstrain_names = eigenstrain
    displacements = 'disp_x disp_y'
  [../]
  [./stress_2]
    type = ComputeLinearElasticStress
    base_name = C_eta2
  [../]
  [./eigenstrain_2]
    type = ComputeEigenstrain
    base_name = C_eta2
    eigen_base = '0.0' #'0.1 0.05 0 0 0 0.01' # The difference is zero V_BCT - V_BCT = 0.0
    #prefactor = -1 #pre # -1
    prefactor = -1.0E-03 #pre # -1 For multiplying strain by 0 to account for C_ijkl_scaled = 1.0E+09C_ijkl (GPa to Pa), and the scaling (es/ls^3) 
    eigenstrain_name = eigenstrain
  [../]
#### New Addition
    [./pre]
      type = GenericConstantMaterial
      prop_names = pre
      #prop_values = 0.02
      prop_values = 0.002
    [../]



# 
   [./fel_eta2]      
  type = ElasticEnergyMaterial
  args = ' '
  base_name = C_eta2
  f_name = fel2
  output_properties = fel2
  outputs = exodus
[../]
################################################
# Sum of Chemical, Electromigration and Elastic Energy
#######################################################3
##################################
#sum chemical and elastic energies
# fel1 for LI2SN5 phase
# fel2 is for BCT matrix
# Therefore, three ElasticEnergyMaterial type are sufficient, 6 is redundant for code cleaning in future works.
[./F_1]
  type = DerivativeSumMaterial
  f_name = F1
  args = 'c1a'
  sum_materials = 'f1chemem fel1' #'Fch1 fel1'
  #sum_materials = 'fch0'
  outputs = exodus
[../]
[./F_2]
  type = DerivativeSumMaterial
  f_name = F2
  args = 'c2a'
  sum_materials = 'f2chemem fel2'
  #sum_materials = 'fch1'
  outputs = exodus
[../]

############################################
# base name of all ComputeElasticityTensor, ComputeSmallStrain, ComputeLinearElasticStress, ComputeEigenstrain should be same
# They are fundamentally the base codes
# The calculation presumably goes to MultiPhaseStressMaterial.
################################################################
# Generate global stress from the phase stresses
  # [./combined]  # replace it with global_stress. SAME thing
  # [./combined]
  #   type = MultiPhaseStressMaterial
  #   phase_base = 'C_eta1  C_eta1  C_eta345 C_eta345 C_eta345 C_eta6'
  #   # phase_base = 'C_eta1  C_eta1  C_eta1 C_eta1 C_eta1 C_eta1'
  #   h          = 'h1 h2 h3 h4 h5 h6'
  #   base_name = global
  # [../]
    [./global_stress]
      type = MultiPhaseStressMaterial
      phase_base = 'C_eta1  C_eta2'
      # phase_base = 'C_eta1  C_eta1  C_eta1 C_eta1 C_eta1 C_eta1'
      h          = 'h1 h2'
      base_name = global
    [../]
[./global_strain]
  type = ComputeSmallStrain
  displacements = 'disp_x disp_y'
[../]
###############################################33


 
############################################
#SwitchingFunction        ## Eq 10,11 of https://doi.org/10.1016/j.actamat.2010.10.038 A quantitative and thermodynamically Moelans 2011
   [./h1]
        type = SwitchingFunctionMultiPhaseMaterial
        h_name = h1
        all_etas = 'eta1 eta2'
        phase_etas = eta1
        #outputs = exodus
    [../]
######################
######################

######################
   [./h2]
        type = SwitchingFunctionMultiPhaseMaterial
        h_name = h2
        all_etas = 'eta1 eta2'
        phase_etas = eta2
        #outputs = exodus
    [../]
######################


############################################    
    # Barrier functions for each phase
  [./g1]
    type = BarrierFunctionMaterial
    g_order = SIMPLE
    eta = eta1
    function_name = g1
  [../]
######################

######################
  [./g2]
    type = BarrierFunctionMaterial
    g_order = SIMPLE
    eta = eta2
    function_name = g2
  [../]
######################



############################################
# constant properties M is needed by SplitCHWRes kernel

  [./constants_mobility] #maximum limit for convergence 1.0E-17 
    type = GenericConstantMaterial
    prop_names  = 'pseudo_L_si   pseudo_kappa  D  M_si_li2sn5  M_si_bct  pseudo_mu'
    prop_values = '50   0.5   1  2.5e-22 2.5e-20 500 ' # L must be 10 or greater (Eureka for convergence test) Ratio between fbulk to L, M, m and kappa important
    #prop_values = '50   0.5   1  5.028e-22 5.028e-21 5.028e-20 500 ' # L must be 10 or greater (Eureka for convergence test) Ratio between fbulk to L, M, m and kappa important
    #prop_values = '50   0.5   1  1.0e-21 1.0e-20 1.0e-19  500 ' # L must be 10 or greater (Eureka for convergence test) Ratio between fbulk to L, M, m and kappa important
    #prop_values = '50   0.5   1  1.0e-20 1.0e-19 1.0e-18  500 ' # L must be 10 or greater (Eureka for convergence test) Ratio between fbulk to L, M, m and kappa important
    #prop_values = '50   0.5   1  1.0e-20 1.0e-19 1.0e-18  500 ' # L must be 10 or greater (Eureka for convergence test) Ratio between fbulk to L, M, m and kappa important
    #prop_values = '50   0.5   1  1.0e-29 1.0e-28 1.0e-27  500 ' # L must be 10 or greater (Eureka for convergence test) Ratio between fbulk to L, M, m and kappa important
    #prop_values = '50   0.5   1  1.0e-28 1.0e-27 1.0e-26  500 ' # L must be 10 or greater (Eureka for convergence test) Ratio between fbulk to L, M, m and kappa important
    #prop_values = '50   0.5   1  1.0e-20 1.0e-19 1.0e-18  500 ' # L must be 10 or greater (Eureka for convergence test) Ratio between fbulk to L, M, m and kappa important
    #prop_values = '50   0.5   1  1.0e-19  500 ' # L must be 10 or greater (Eureka for convergence test) Ratio between fbulk to L, M, m and kappa important
    #prop_values = '10  0.5   1  8  1.5 500 ' #Perfect for factor_fi=1.0
    #prop_values = '1000 0.5   1  800 1.5 1000' #perfect
    #prop_values = '100.0 0.5   1  80.0 1.5 1000'
    #prop_values = '50.0 0.5   1  40.0 1.5 500' # converged
    #prop_values = '10.0 0.5   1  10.0 1.5 1000' #solve converged once
    #prop_values = '1.0 1.0    1  1.0 1.5 500' #more more better
    #prop_values = '1.0 1.0    1  1.0 1.5 150' #more more better
    #prop_values = '1.0 5.0    1  1.0 1.5 0.5' # more better
    #prop_values = '0.17 5.0    1  0.1 1.5 0.5' #better
    #prop_values = '0.0017 0.005    1  0.001 1.5 0.05' # no convergence
    #prop_values = '0.017 5.0    1  0.01 1.5 0.5' # good
  [../]
############################################
#####44.45 nm is the least thickness of delta for 6.67 criterion
#####making delta = 500 nm brought convergence at dt = 100 ns
[./constants_interface]
    type = GenericConstantMaterial
    prop_names  = 'gamma sigma delta length_scale energy_scale time_scale'
    prop_values = '1.5 0.5   75E-09  1e9 6.24150943e18 1.0e9 ' # L must be 10 or greater (Eureka for convergence test) Ratio between fbulk to L, M, m and kappa important
    #prop_values = '1.5 0.5   35E-09  1e9 6.24150943e18 1.0e9 ' # L must be 10 or greater (Eureka for convergence test) Ratio between fbulk to L, M, m and kappa important
    #prop_values = '1.5 0.5   55E-09  1e9 6.24150943e18 1.0e9 ' # L must be 10 or greater (Eureka for convergence test) Ratio between fbulk to L, M, m and kappa important
    #prop_values = '1.5 0.5   25E-09  1e9 6.24150943e18 1.0e9 ' # L must be 10 or greater (Eureka for convergence test) Ratio between fbulk to L, M, m and kappa important
    #prop_values = '1.5 0.5   40E-09  1e9 6.24150943e18 1.0e9 ' # L must be 10 or greater (Eureka for convergence test) Ratio between fbulk to L, M, m and kappa important
[../]
  ################################################33
[./kappa]
    type = ParsedMaterial
    material_property_names = 'sigma delta length_scale energy_scale'
    f_name = kappa
    constant_names = 'factor_kappa' # factor_kappa = factor_sigma*factor_delta = 2*10  (works at dt = 1000), 2*8=16.0 also works; 1.5*15 = 22.5 too works, 2*1000 = 2000 works
    constant_expressions = '1000' #'10' #'4' #'100' #'20' #'18.0' #'22.50' # 1.0E-1 to make the kappa scaled from 702.225 to 70.225 so that its magnitude is less than scaled fbulk's absolute magnitude
    function = '(energy_scale/length_scale)*0.75*sigma*delta*factor_kappa' #eV/nm
  [../]
  [./mu]
    type = ParsedMaterial
    material_property_names = 'sigma delta length_scale energy_scale'
    f_name = mu
    constant_names = 'factor_mu' #factor_mu = factor_sigma/factor_delta = 2/10 = 0.2 (works at dt = 1000), 2/8 = 0.25 also works, 1.5/15=0.1 too works, 2/1000=0.002 works
    constant_expressions = '10' #'2' #'2.5' #'1.0' #'0.2' #'0.05' #'8.0' #'4.80E+00' #'1.0E-01'
    function = '(energy_scale/(length_scale)^3)*6*(sigma/delta)*factor_mu' #eV/nm^3
  [../]
############################################
 [./interface_mobility] # considered the same in isotropic and anisotropic
    type = ParsedMaterial
    #material_property_names = 'M_si mu kappa'
    f_name = L
    constant_names = 'factor_L'
    constant_expressions = '1.0E+02' # To make the scaled quantity in the range of 5
    material_property_names = 'length_scale energy_scale time_scale M mu kappa' # We will later use M_si instead of M because of unit reasons
    function = '((length_scale)^3/(energy_scale*time_scale))*(16/3)*(mu*M/kappa)*factor_L' #l^3/energy*time
  [../]


  [./ch_mobility] # considered the same in isotropic and anisotropic
    type = ParsedMaterial
    #material_property_names = 'M_si'
    f_name = M
    #args = 'eta1 eta2'
    material_property_names = 'length_scale energy_scale time_scale M_si_li2sn5 M_si_bct  M_gb h1 h2 '
    function = '((length_scale)^5/(energy_scale*time_scale))*((h1)*M_si_li2sn5+h2*M_si_bct+(h2)*(h2)*M_gb)' #l^5/energy*time
  [../]
  
  [./M_gb]
    type = ParsedMaterial
    material_property_names = 'M_si_li2sn5'
    f_name = M_gb
    function = '5*M_si_li2sn5'
    #function = '25*M_si_imc'
    #function = '35*M_si_imc' # 4 non-convergence to reach 1st convergence
    #function = '50*M_si_imc' #  4 non-convergence to reach 1st convergence
    #function = '20*M_si_imc' # 3 non-convergence to reach 1st convergence, compared to Mgb=200Mimc, this model converges well and GB is resolves better earlier owing to slow growth 
    #function = '200*M_si_imc'
    #function = '400*M_si_imc'# compared to Mgb=200Mimc, this model convergence is difficult and GB is not resolved better  owing to faster growth 
  [../]
  
###########################################
  # Electromigration properties
  [./electromigration_properties] #optimized z
      type = GenericConstantMaterial
      prop_names = 'Vmimc Nav echarge zeff1 zeff2 pseudoelectcond'
      prop_values = '16.29E-06 6.0221E+23 -1.6022E-19 6.63 34.095  5.714E+06' #m to nm J to eV s to h the univt of 16.29E-06 m3/mol
      #prop_values = '16.29E-06 6.0221E+23 -1.6022E-19 0 0 0 5.714E+06' #m to nm J to eV s to h the univt of 16.29E-06 m3/mol
  [../]
  
  [./electriccond] # considered the same in isotropic and anisotropic
    type = ParsedMaterial
    f_name = electcond
    #args = 'eta1 eta2 eta3 eta4 eta5 eta6 eta7 eta8 eta9 eta10 eta11 eta12'
    constant_names = 'factor_ec ec_1 ec_2' # Use Equation 25 of Hektor2016 paper M=sigma_hiMi, where Mi=D_i/A_i and A_i = coeff of parabolic free energy
    constant_expressions = '1.0 5.714E+06 9.096E+06 ' #m^2 mol/J s mobility of Sn is used as 10E-23 to be near to that of Cu and IMC
    material_property_names = 'length_scale energy_scale time_scale  h1 h2'
    function = 'factor_ec*(1.0/(length_scale))*(5.714E+06)'
    #function = 'factor_ec*(1.0/(length_scale))*(ec_1*h1+ec_2*(h2+h4+h5+h6+h7+h8+h9+h10+h11+h12)+ec_3*h3)'
  [../]
##############################################################################################
[]

########################################################################################



########################################################################################

[Kernels]

############################################
######## First put the KKS condition with kernels of phase concentrations (local) related to cA global #######

############################################
# Phase concentration constraints
  [./chempot12a]
    type = KKSPhaseChemicalPotential
    variable = c1a
    cb       = c2a
    fa_name  = F1
    fb_name  = F2
  [../]
######################

######################
#  [./chempot23a]
#    type = KKSPhaseChemicalPotential
#    variable = c2a
#    cb       = c3a
#    fa_name  = F2
#    fb_name  = F3
#  [../]
############################################

############################################
  [./phaseconcentration_a]
    type = KKSMultiPhaseConcentration
    variable = c2a
    cj = 'c1a c2a '
    hj_names = 'h1 h2'
    etas = 'eta1 eta2'
    c = ca
  [../]
############################################

############################################
######## First put the KKS condition for kernels of phase concentrations (local) related to cB global #######

############################################
# Phase concentration constraints
#  [./chempot12b]
#    type = KKSPhaseChemicalPotential
#    variable = c1b
#    cb       = c2b
#    fa_name  = F1
#    fb_name  = F2
#  [../]
######################

######################
#  [./chempot23b]
#    type = KKSPhaseChemicalPotential
#    variable = c2b
#    cb       = c3b
#    fa_name  = F2
#    fb_name  = F3
#  [../]
############################################

############################################
#  [./phaseconcentration_b]
#    type = KKSMultiPhaseConcentration
#    variable = c3b
#    cj = 'c1b c2b c3b c2b c2b'
#    hj_names = 'h1 h2 h3 h4 h5'
#    etas = 'eta1 eta2 eta3 eta4 eta5'
#    c = cb
#  [../]
############################################

############################################
## Kernels for split Cahn-Hilliard type equation
    ## CHBulk known as KKSSplitCHCRes is here to replace SplitCHParsed
    ## because in KKS model , gradient energy term is not allowed in the C-H type equation [Tonks2018-ComputationalMaterialsScience,vol. 147, pp.353-362.]
    ## while SplitCHParsed kernel consists of the term k\nabla^2 c_i (thus making it unsuitable here), KKSSplitCHCRes fortunately avoids this term.
    ## Never use SplitCHParsed kernel with KKS model
    ## Because of the KKS condition 1 (equality of chemical potential between any two adjacent phases), one KKSSplitCHCRes kernel (either for c1, c2 or c3) is sufficient and there is no need to put three such kernels corresponding to c1, c2 and c3.

##############################################################################3  
# Diffusion kernels corresponding to phase concentrations of global cA
########################################################################  
    [./CHBulka] # Gives the residual for the concentration, dF/dc-mu
        type = KKSSplitCHCRes
        variable = ca
        ca       = c2a
        fa_name  = F2 #only F2 is used
        w        = wa
    [../]
############################################

############################################
    [./dcdta] # Gives dc/dt
        type = CoupledTimeDerivative
        variable = wa
        v = ca
    [../]
############################################

############################################    
    [./ckernela] # Gives residual for chemical potential dc/dt+M\grad(mu)
        type = SplitCHWRes
        mob_name = M
        variable = wa
        args = 'eta1 eta2'
    [../]
############################################

##############################################################################3  
# Diffusion kernels corresponding to phase concentrations of global cB
########################################################################  
#    [./CHBulkb] # Gives the residual for the concentration, dF/dc-mu
#        type = KKSSplitCHCRes
#        variable = cb
#        ca       = c2b
#        fa_name  = F2 #only F2 is used
#        w        = wb
#    [../]
############################################

############################################
#    [./dcdtb] # Gives dc/dt
#        type = CoupledTimeDerivative
#        variable = wb
#        v = cb
#    [../]
############################################

############################################    
#    [./ckernelb] # Gives residual for chemical potential dc/dt+M\grad(mu)
#        type = SplitCHWRes
#        mob_name = M
#        variable = wb
#        args = 'eta1 eta2 eta3 eta4 eta5 eta6'
#    [../]
############################################

##################################################################

  # Kernels for Allen-Cahn equation for eta1
  
######################  
  [./deta1dt]
    type = TimeDerivative
    variable = eta1
  [../]
######################  
######################################################################################################################################
###################################################################################################################
  [./ACBulkF1]
    type = KKSMultiACBulkF
    variable  = eta1
    Fj_names  = 'F1 F2'
    hj_names  = 'h1 h2'
    gi_name   = g1
    eta_i     = eta1
    wi        = 5.0 #1.0
    args      = 'c1a c2a eta2'
    mob_name = L
  [../]
######################
###############################################################################################################
# In ACBulkC1, the kernel requires  the definition of a particular element (A or B) as a separate list of vectors
################################################################################################################
######################  
  [./ACBulkC1a]
    type = KKSMultiACBulkC
    variable  = eta1
    Fj_names  = 'F1 F2'
    hj_names  = 'h1 h2 '
    cj_names  = 'c1a c2a'
    eta_i     = eta1
    args      = 'eta2'
    mob_name = L
  [../]
######################
 #[./ACBulkC1b]
 #   type = KKSMultiACBulkC
 #   variable  = eta1
 #   Fj_names  = 'F1 F2'
 #   hj_names  = 'h1 h2 '
 #   cj_names  = 'c1b c2b c3b c2b c2b'
 #   eta_i     = eta1
 #   args      = 'eta2 eta3 eta4 eta5 '
 #   mob_name = L
 # [../]

######################################################################################################################################
######################  
  [./ACInterface1]
    type = ACInterface
    variable = eta1
    kappa_name = kappa
    mob_name = L
  [../]
##################################################################
##################################################################
# This kernel requires the model parameter m (mu) and the gamma parameter
###########################################################################
[./ACdfintdeta1] #L*m*(eta_i^3-eta_i+2*beta*eta_i*sum_j eta_j^2)
      type = ACGrGrMulti
      variable = eta1
      v = 'eta2'
      gamma_names = 'gamma'
      mob_name = L
      args = 'eta2'
[../]
 
##################################################################

  # Kernels for Allen-Cahn equation for eta2
  
######################
  [./deta2dt]
    type = TimeDerivative
    variable = eta2
  [../]
######################

######################  
  [./ACBulkF2]
    type = KKSMultiACBulkF
    variable  = eta2
    Fj_names  = 'F1 F2'
    hj_names  = 'h1 h2'
    gi_name   = g2
    eta_i     = eta2
    wi        = 5.0 #1.0
    args      = 'c1a c2a eta1'
    mob_name = L
  [../]
######################

######################
  [./ACBulkC2a]
    type = KKSMultiACBulkC
    variable  = eta2
    Fj_names  = 'F1 F2'
    hj_names  = 'h1 h2'
    cj_names  = 'c1a c2a'
    eta_i     = eta2
    args      = 'eta1'
    mob_name = L
  [../]
######################
# [./ACBulkC2b]
#    type = KKSMultiACBulkC
#    variable  = eta2
#    Fj_names  = 'F1 F2 F2 F2 F2'
#    hj_names  = 'h1 h2 h3 h4 h5'
#    cj_names  = 'c1b c2b c3b c2b c2b'
#    eta_i     = eta2
#    args      = 'eta1 eta3 eta4 eta5'
#    mob_name = L
#  [../]

######################  
  [./ACInterface2]
    type = ACInterface
    variable = eta2
    kappa_name = kappa
    mob_name = L
  [../]
##################################################################
# This kernel requires the model parameter m (mu) and the gamma parameter
#################################################################
[./ACdfintdeta2] #L*m*(eta_i^3-eta_i+2*beta*eta_i*sum_j eta_j^2)
      type = ACGrGrMulti
      variable = eta2
      v = 'eta1'
      gamma_names = 'gamma'
      mob_name = L
      args = 'eta1'
[../]
#################################################################
########################################################################################
####Laplacian of potential variable
# Kernel for electric potential
  [./Laplacian]
    type = MatDiffusion
    variable = pot
    diffusivity = electcond # electrical_conductivity name in Derivative Parsed Material parser , sigma_el(eta_i)
  [../]


########################################################################################
####################################################################################################################################
# For elasticity
# Kernel for NS

  [./TensorMechanics]
   displacements = 'disp_x disp_y'
   base_name=global
   # Two new lines for plane strain formulation
   planar_formulation = PLANE_STRAIN 
   use_displaced_mesh = false
  [../]
########################################################################################



[]

########################################################################################




########################################################################################

[AuxKernels]
[./bnds]
    type = BndsCalcAux
    variable = bnds
    var_name_base = eta
    op_num = 2 #2
    v = 'eta1 eta2' #Not writing a variable here will put a 0 value on the eta value of the absentee
  [../]
############################################
  [./Energy_total]
    type = KKSMultiFreeEnergy
    Fj_names = 'F1 F2'
    hj_names = 'h1 h2'
    gj_names = 'g1 g2'
    variable = Energy
    w = 1
    interfacial_vars =  'eta1  eta2'
    kappa_names =       'kappa kappa'
  [../]
############################################
############################################
[./grains_hsquarec]
      type = TwoPhasesSumCdothsquare
      variable = gr_c
      var1 = ca
      h1_name = h1
      h2_name = h2
[../]
#################################################
# # Stress due to elasticity

[./von_mises_kernel]
  #Calculates the von mises stress and assigns it to von_mises
  type = RankTwoScalarAux
  variable = von_mises
  rank_two_tensor =global_stress
  execute_on = timestep_end
  scalar_type = VonMisesStress #TODO: Check units
[../]
[./matl_sigma11]
  type = RankTwoAux
  rank_two_tensor = global_stress
  index_i = 0
  index_j = 0
  variable = sigma11
[../]
[./matl_sigma22]
  type = RankTwoAux
  rank_two_tensor = global_stress
  index_i = 1
  index_j = 1
  variable = sigma22
[../]
[./matl_sigma12]
  type = RankTwoAux
  rank_two_tensor = global_stress
  index_i = 0
  index_j = 1
  variable = sigma12
[../]
[./matl_e11]
  type = RankTwoAux
  rank_two_tensor = total_strain
  index_i = 0
  index_j = 0
  variable = e11
[../]
[./matl_e12]
  type = RankTwoAux
  rank_two_tensor = total_strain
  index_i = 0
  index_j = 1
  variable = e12
[../]
[./matl_e22]
  type = RankTwoAux
  rank_two_tensor = total_strain
  index_i = 1
  index_j = 1
  variable = e22
[../]
[]

########################################################################################

########################################################################################
[Postprocessors]
    [./eta1_area_h]
      type = ElementIntegralMaterialProperty
      mat_prop = h1
      execute_on = 'Initial TIMESTEP_END'
    [../]
    [./eta2_area_h]
      type = ElementIntegralMaterialProperty
      mat_prop = h2
      execute_on = 'Initial TIMESTEP_END'
    [../]
   
[]

########################################################################################

[Executioner]

############################################
#  type = Transient
#  solve_type = 'PJFNK'
#  petsc_options_iname = '-pc_type -sub_pc_type   -sub_pc_factor_shift_type'
#  petsc_options_value = 'asm       ilu            nonzero'
#  l_max_its = 100 #100 #30
#  nl_max_its = 50 #10
#  l_tol = 1.0e-4
#  nl_rel_tol = 1.0e-9 #1.0e-10
#  nl_abs_tol = 1.0e-10 #1.0e-11

#  #num_steps = 45 #2
#  #dt = 0.5
#  end_time=1.0E+010
  ############################################
#[./TimeStepper]
 #   ## Turn on time stepping
#    type = IterationAdaptiveDT
#    dt = 1.0E+01 #-02 #1.0E-02 #1.0E-1 #0.5 #1.00E+04 #06
#    cutback_factor = 0.7
#    growth_factor = 1.50 #1.5
#    optimal_iterations = 7
#    #num_steps = 55
#[../]
############################################
############################################
  type = Transient
  solve_type = 'PJFNK'
  petsc_options_iname = '-pc_type -sub_pc_type   -sub_pc_factor_shift_type'
  petsc_options_value = 'asm       ilu            nonzero'
  l_max_its = 100 #100 #30
  nl_max_its = 50 #10
  l_tol = 1.0e-4
  nl_rel_tol = 1.0e-8 # 1.0e-10
  nl_abs_tol = 1.0e-9 # 1.0e-11

  #num_steps = 45 #2
  #dt = 0.5
  dtmax = 100000 #2500 #5000 #10000
  dtmin = 1 #250 #500
  end_time=1.0E+012 #10
  ############################################
[./TimeStepper]
    ## Turn on time stepping
    type = IterationAdaptiveDT
    dt = 1E+02 #100 #500# 10 #100.0 #100 #200 #225 #1250 #100 #25 #250 #850 #1250 # 100 #250 #500 #750 #1000.0E+00 #5.0e-03  #1.0e+05 #best is 1.0e+05 # 1.0e+03 #1.0
    #dt = 10.0 #1.0E+01 #-02 #1.0E-02 #1.0E-1 #0.5 #1.00E+04 #06
    cutback_factor = 0.85 #0.95 #0.9999 #0.7 #0.9999 #0.7 # 0.9999 #0.7 # 0.9999 #0.9999 #0.95 #0.9 #0.7 #0.9999 #0.9 #0.7 #0.7
    growth_factor = 1.25 #1.5 # 1.0001 #1.5 # 1.0001 #1.5  #1.5 #1.50 #1.5
    optimal_iterations = 7 #2 #7
    #num_steps = 55
[../]
############################################
############################################
# adaptive mesh to resolve an interface
#   [./Adaptivity]
#     initial_adaptivity    = 3 #3 #3 #2 #3 #3 #2             # Number of times mesh is adapted to initial condition
#     refine_fraction       = 0.7 #0.7           # Fraction of high error that will be refined
#     coarsen_fraction      = 0.1           # Fraction of low error that will coarsened
#     max_h_level           = 2 #4 #3 #4 #4 #3 #2 #3 #3 #2 #3             # Max number of refinements used, starting from initial mesh (before uniform refinement)
#     weight_names          = 'eta1 eta2'
#    weight_values         = '1 1'
#   [../]
[]

########################################################################################



########################################################################################

[Preconditioning]

############################################
  active = 'full'
  [./full]
    type = SMP
    full = true
  [../]
  [./mydebug]
    type = FDP
    full = true
  [../]
############################################

[]

########################################################################################



########################################################################################

[Outputs]

############################################
  exodus = true
  csv  = true
  interval = 1 #50 #1 #50 #1 #50 #10 #20 #100 #100 #1 #20 #10 #5 #1 #10 #5 #50 #1 #50 #20 #10
  checkpoint = true
############################################
 #[./my_checkpoint]
  #  type = Checkpoint
  #  num_files = 4
  #  interval = 10 #5
 # [../]

[]

########################################################################################


[Debug]
  show_var_residual_norms = true
[]
