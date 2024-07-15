import os
import numpy as np
import utils

def main(u_size, n_size, gslinea, pCH4FlowZ, pGasPressLPZ):
    z_size = pCH4FlowZ.shape[3]
    pCard_u = utils.calc_pCard(gslinea, pGasPressLPZ, z_size)
    pCard_uu = utils.calc_pCard(gslinea, pCH4FlowZ, z_size)
    pAllComb, pPreCalc_IEQ3, pPreCalc_IEQ4 = utils.init_pAllComb(
        pGasPressLPZ=pGasPressLPZ,
        pCH4FlowZ=pCH4FlowZ,
        gslinea=gslinea,
        u_size=u_size,
        pCard_u=pCard_u,
        pCard_uu=pCard_uu
    )

    pAllComb, pMultRHS = utils.calc_FilterIndexes(
        pAllComb=pAllComb, gslinea=gslinea, pCard_u=pCard_u, pCard_uu=pCard_uu)

    # Create output folders
    os.makedirs("outputs/", exist_ok=True)


    dim_names = ["gsi", "gsj", "gsc", "u", "uu", "z"]
    utils.export("outputs/pPreCalc_IEQ3", pPreCalc_IEQ3)
    utils.export("outputs/pPreCalc_IEQ4", pPreCalc_IEQ4)
    utils.export("outputs/pAllComb", pAllComb, dim_names)

    pMultRHSNeg = np.zeros(pMultRHS.shape)
    pMultRHSPos = np.zeros(pMultRHS.shape)
    pMultRHSNeg[pMultRHS < 0] = pMultRHS[pMultRHS < 0]
    pMultRHSPos[pMultRHS > 0] = pMultRHS[pMultRHS > 0]

    utils.export("outputs/pMultRHSNeg", pMultRHSNeg, dim_names)
    utils.export("outputs/pMultRHSPos", pMultRHSPos, dim_names)

    pRHSAuxNeg, pRHSAuxPos = utils.set_pMultRHS(pMultRHS, n_size)

    utils.export("outputs/pRHSAuxNeg", pRHSAuxNeg, dim_names + ["n"])
    utils.export("outputs/pRHSAuxPos", pRHSAuxPos, dim_names + ["n"])

    # [6.6] Calculate pRHSAux (Z-Aux_l_u_v_z_w)
    pRHSAux = pRHSAuxNeg + pRHSAuxPos

    # [7.1] Calculate pRHSNeg and pRHSPos
    pRHSNeg = np.sum(pRHSAuxNeg, axis=-2) // z_size  
    pRHSPos = np.sum(pRHSAuxPos, axis=-2) // z_size

    utils.export("outputs/pRHSNeg", pRHSNeg, dim_names)
    utils.export("outputs/pRHSPos", pRHSPos, dim_names)

    # [7.2] Calculate pRHS_sign (Z-sgn_l_u_v_w) and pRHS (Z-RHS_l_u_v_w)
    pRHS_sign = pRHSNeg + pRHSPos
    pRHS = -np.abs(pRHS_sign)

    utils.export("outputs/pRHS", pRHS, dim_names)
    utils.export("outputs/pRHSAux", pRHSAux, dim_names + ["n"])

    pPreCalcAux = utils.calc_pPreCalcAux(pAllComb, gslinea, n_size)
    utils.export("outputs/pPreCalcAux", pPreCalcAux, dim_names + ["n"])

    #[8.2] Calculate pPreCalc (Z-Pre_l_u_v_w)
    pPreCalc = np.zeros(pPreCalcAux.shape)
    pPreCalc[pRHSAux != 0] = pPreCalcAux[pRHSAux != 0]
    utils.export("outputs/pPreCalc", pPreCalc, dim_names + ["n"])

#--------------------------------------------------
#               Input data
#--------------------------------------------------
if __name__ == "__main__":
    # Number of grid points for z linearization
    z_values = np.array([z for z in range(1, 7)])
    # [2] u = v = 1,2,... max(F_gslinea_|z| - F_gslinea_ord(z=2), P_gslinea_|z| - P_gslinea_ord(z=2))
    u_values = np.array([u for u in range(1, 707)])
    # [6.1] n_values = 1,2,...round off (|z|-1 / 2)
    n_values = np.array([n for n in range(1, 4)])
    # Gas nodes
    gsi_values = np.array([gsi for gsi in range(1, 3)])
    # Indentifier for parallel pipelines connection two nodes
    gsc_values = np.array([1])
    # Pipelines
    gslinea = np.array([[0, 1, 0]])

    values = [gsi_values, gsi_values, gsc_values, z_values]
    pCH4FlowZ = np.zeros([
        len(gsi_values),
        len(gsi_values),
        len(gsc_values),
        len(z_values)
    ], dtype=np.int64)

    # Average gas flow at grid point z 
    pCH4FlowZ[0, 1, 0, 0] = 0
    pCH4FlowZ[0, 1, 0, 1] = 152
    pCH4FlowZ[0, 1, 0, 2] = 266
    pCH4FlowZ[0, 1, 0, 3] = 476
    pCH4FlowZ[0, 1, 0, 4] = 634
    pCH4FlowZ[0, 1, 0, 5] = 858

    pGasPressLPZ = np.zeros([
        len(gsi_values),
        len(gsi_values),
        len(gsc_values),
        len(z_values),
    ], dtype=np.int64)

    # Corresponding pressure difference at gird point z
    pGasPressLPZ[0, 1, 0, 0] = 0
    pGasPressLPZ[0, 1, 0, 1] = 1
    pGasPressLPZ[0, 1, 0, 2] = 3
    pGasPressLPZ[0, 1, 0, 3] = 9
    pGasPressLPZ[0, 1, 0, 4] = 15
    pGasPressLPZ[0, 1, 0, 5] = 25

    with utils.Timer() as _:
        main(
            u_size=len(u_values),
            n_size=len(n_values),
            gslinea=gslinea,
            pCH4FlowZ=pCH4FlowZ,
            pGasPressLPZ=pGasPressLPZ
        )
