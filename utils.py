import numpy as np
import numba
import datetime

class Timer:
    def __enter__(self):
        self.t = datetime.datetime.now()

    def __exit__(self, exc_type, exc_val, exc_tb):
        t = datetime.datetime.now()
        delta = t - self.t
        print(delta)



def to2DArray(arr):
    shape = arr.shape
    indices = np.indices(shape)
    reshaped_indices = indices.reshape(len(shape), -1).T
    flattened_values = arr.flatten()
    result = np.column_stack((reshaped_indices, flattened_values))
    return result


def export(fileName, input_arr, *dims):
    """
    exports the content of multidimensional array to file
    :param fileName: str
    :param arr: numpy array
    :param dims: list[string], optional dimension names
    :return: unit
    """

    if not dims:
        with open(fileName,"w+") as f:
            for coords in np.ndindex(input_arr.shape):
                val = input_arr[coords]
                if val == 0:
                    continue
                coords_str = "".join(f"{x} " for x in coords)
                line = coords_str + f" {val}"
                f.write(line)
    else:
        dims = dims[0]
        with open(fileName, "w+") as f:
            for coords in np.ndindex(input_arr.shape):
                val = input_arr[coords]
                if val == 0:
                    continue

                line = ""
                for k, v in zip(dims, coords):
                    line += "{}_{:.0f} ".format(k, v+1)

                f.write(f"{line} {val}\n")


@numba.njit(cache=True)
def check_ratio(u, uu, w, ww):
    return ((w + 1) / (u + 1)) == ((ww + 1) / (uu + 1))

@numba.njit(cache=True)
def init_pAllComb(pGasPressLPZ, pCH4FlowZ, gslinea, u_size, pCard_u, pCard_uu):
    pGasPress_GCD = pGasPressLPZ.copy()
    pGasFlow_GCD = pCH4FlowZ.copy()
    gsi_size = pCH4FlowZ.shape[0]
    gsc_size = pCH4FlowZ.shape[2]
    z_size = pCH4FlowZ.shape[3]

    pPreCalc_IEQ3 = np.zeros((gsi_size, gsi_size, gsc_size, z_size, z_size))
    pPreCalc_IEQ4 = pPreCalc_IEQ3.copy()
    pAllComb = np.zeros((gsi_size, gsi_size, gsc_size, u_size, u_size, z_size))
    pGCD = np.zeros((gsi_size, gsi_size, gsc_size, z_size))

    pCH4Flow = np.arange(1, u_size + 1)
    pGasPress = np.arange(1, u_size + 1)

    for gsi, gsj, gsc in gslinea:
        for z in numba.prange(z_size):
            a = pGasPress_GCD[gsi, gsj, gsc, z]
            b = pGasFlow_GCD[gsi, gsj, gsc, z]

            for zz in range(z_size):
                # [1.1] A = min (...)
                if z != zz:
                    pCH4FlowZ_z = pCH4FlowZ[gsi, gsj, gsc, z]
                    pCH4FlowZ_zz = pCH4FlowZ[gsi, gsj, gsc, zz]
                    pGasPressLPZ_z = pGasPressLPZ[gsi, gsj, gsc, z]
                    pGasPressLPZ_zz = pGasPressLPZ[gsi, gsj, gsc, zz]

                    pPreCalc_IEQ4[gsi, gsj, gsc, z, zz] = \
                        (pGasPressLPZ_z * pCH4FlowZ_zz) - (pCH4FlowZ_z * pGasPressLPZ_zz)

                # [1.2] B = max (...) and C = min (...)
                if 0 < z <= zz:
                    pCH4FlowZ_z = pCH4FlowZ[gsi, gsj, gsc, z]
                    pCH4FlowZ_zz = pCH4FlowZ[gsi, gsj, gsc, zz]
                    pGasPressLPZ_z = pGasPressLPZ[gsi, gsj, gsc, z]
                    pGasPressLPZ_zz = pGasPressLPZ[gsi, gsj, gsc, zz]

                    pPreCalc_IEQ3[gsi, gsj, gsc, z, zz] = \
                        (pCH4FlowZ_z * pGasPressLPZ_zz) - (pGasPressLPZ_z * pCH4FlowZ_zz)
                    
            pGCD[gsi, gsj, gsc, z] = np.gcd(a, b)
            # [3] Calculate pAllComb (Z_l_u_v_z)
            pCard_u_val = pCard_u[gsi, gsj, gsc]
            pCard_uu_val = pCard_uu[gsi, gsj, gsc]
            for u in range(pCard_u_val):
                for uu in range(pCard_uu_val):
                    a = pCH4Flow[u] * pCH4FlowZ[gsi, gsj, gsc, z]
                    b = pGasPress[uu] * pGasPressLPZ[gsi, gsj, gsc, z]
                    pAllComb[gsi, gsj, gsc, u, uu, z] = a - b



    return pAllComb, pPreCalc_IEQ3, pPreCalc_IEQ4

@numba.njit(parallel=True, cache=True)
def calc_FilterIndexes(pAllComb, gslinea, pCard_u, pCard_uu):
    u_size = pAllComb.shape[3]
    z_size = pAllComb.shape[5]

    pMultRHS = np.zeros(pAllComb.shape, dtype=np.int64)
    for (gsi, gsj, gsc) in gslinea:
        pCard_u_val = pCard_u[gsi, gsj, gsc]
        pCard_uu_val = pCard_uu[gsi, gsj, gsc]

        for u in range(pCard_u_val):  # switch u_size for pCard
            for uu in range(pCard_uu_val):
                for w in range(pCard_u_val):
                    for ww in range(pCard_uu_val):
                        for z in range(z_size):
                            for zz in range(z_size):
                                pAC_w = pAllComb[gsi, gsj, gsc, w, ww, z]
                                pAC_u = pAllComb[gsi, gsj, gsc, u, uu, z]
                                pAC_zz = pAllComb[gsi, gsj, gsc, u, uu, zz]

                                # [4.1] Set all multiples contained in pAllComb (Z_l_u_v_z) to zero
                                # for index combinations where ord(w)>ord(u) and ord(ww)=ord(uu)
                                if w > u and ww == uu and zz != z and pAC_u != 0:
                                    if pAC_w % pAC_u == 0 and pAC_w == pAC_zz and check_ratio(u, uu, w, ww):
                                        pAllComb[gsi, gsj, gsc, w, ww, z] = 0

                                # [4.2] Set all multiples contained in pAllComb [8.3] Calculate pPreCalc (Z-Pre_l_u_v_w) to zero 
                                # for index combinations where ord(w)=ord(u) and ord(ww)>ord(uu)
                                if w == u and ww > uu and z != zz:
                                    if pAC_u and (pAC_w % pAC_u) == 0 and pAC_u == pAC_zz:
                                        if check_ratio(u, uu, w, ww):
                                            pAllComb[gsi, gsj, gsc, w, ww, z] = 0

                                # [4.3] Set all multiples contained in pAllComb (Z_l_u_v_z) to zero 
                                # for index combinations where ord(w)>ord(u) and ord(ww)>ord(uu)
                                if w > u and ww > uu and zz != z:
                                    if pAC_u and pAC_w % pAC_u == 0 and pAC_u == pAC_zz:
                                        if check_ratio(u, uu, w, ww):
                                            pAllComb[gsi, gsj, gsc, w, ww, z] = 0

                                # [5] Determines all index combinations pMultRHS (Z-hat_l_u_v_z) from pAllComb (Z_l_u_v_z)
                                # that contain at least 2 identical values (per row)
                                if z != zz and pAC_u == pAC_zz:
                                    pMultRHS[gsi, gsj, gsc, u, uu, z] = pAC_u

    return pAllComb, pMultRHS



def set_pMultRHS(pMultRHS, n_size):
    gsi_size = pMultRHS.shape[0]
    gsc_size = pMultRHS.shape[2]
    u_size = pMultRHS.shape[3]
    z_size = pMultRHS.shape[5]

    pMultRHSNeg = np.zeros(pMultRHS.shape, dtype=np.int64)
    pMultRHSPos = np.zeros(pMultRHS.shape, dtype=np.int64)

    pRHSAuxNeg = np.zeros((
        gsi_size,
        gsi_size,
        gsc_size,
        u_size,
        u_size,
        z_size,
        n_size), dtype=np.int64)
    pRHSAuxPos = pRHSAuxNeg.copy()
    #[6.2] Split pMultRHS (Z-hat_l_u_v_z) in a positive and negative component 
    pMultRHSNeg[pMultRHS < 0] = pMultRHS[pMultRHS < 0]
    pMultRHSPos[pMultRHS > 0] = pMultRHS[pMultRHS > 0]

    pCnt = 1
    pMaxN = np.floor(z_size / 2)
    while pCnt <= pMaxN:
        #[6.3] Calculate the maximum absolute value per line
        pMult_CntNeg = np.amax(np.abs(pMultRHSNeg), axis=-1)
        pMult_CntPos = np.amax(np.abs(pMultRHSPos), axis=-1)

        #[6.4] Add index n
        for n in range(n_size):
            if n + 1 == pCnt:
                pRHSAuxNeg[..., n] = -1 * pMult_CntNeg[..., np.newaxis]
                pRHSAuxPos[..., n] = pMult_CntPos[..., np.newaxis]

        #[6.5] Filtering
        pMultRHSNeg = np.where(pMultRHSNeg > pRHSAuxNeg[..., pCnt - 1], pMultRHSNeg, 0)
        pMultRHSPos = np.where(pMultRHSPos < pRHSAuxPos[..., pCnt - 1], pMultRHSPos, 0)
        pCnt += 1

    return pRHSAuxNeg, pRHSAuxPos


@numba.njit(cache=True)
def calc_pPreCalcAux(pAllComb, gslinea, n_size):
    #  np.zeros((gsi_size, gsi_size, gsc_size, u_size, u_size, z_size))
    (gsi_size, _, gsc_size, u_size, _, z_size) = pAllComb.shape
    pPreCalcAux = np.zeros((
        gsi_size,
        gsi_size,
        gsc_size,
        u_size,
        u_size,
        z_size,
        n_size
    ), dtype=np.int64)
    # [8.1] Calculate pPreCalcAux (Z_l_u_v_z_w)
    pCnt = 0
    pMaxN = np.floor(z_size / 2)
    while pCnt <= pMaxN:
        for (gsi, gsj, gsc) in gslinea:
            for z in range(z_size):
                for n in range(n_size):
                    if z > 1 and n == pCnt:
                        pPreCalcAux[gsi, gsj, gsc, :, :, z, n] = pAllComb[gsi, gsj, gsc, :, :, z]
        pCnt += 1

    return pPreCalcAux


@numba.njit(cache=True)
def calc_pCard(gslinea, pGasPressLPZ, z_size):
    """
    pCard_u (gsi,gsj,gsc)$[gslinea(gsi,gsj,gsc)] =
    sum[z$[ord(z)=card(z)], pGasPressLPZ(gsi,gsj,gsc,z)] - sum[z$[ord(z)=2], pGasPressLPZ(gsi,gsj,gsc,z)];


    pCard_uu(gsi,gsj,gsc)$[gslinea(gsi,gsj,gsc)] = sum[z$[ord(z)=card(z)], pCH4FlowZ   (gsi,gsj,gsc,z)] - sum[z$[ord(z)=2], pCH4FlowZ   (gsi,gsj,gsc,z)];
    """
    gsi_size = pGasPressLPZ.shape[0]
    gsj_size = pGasPressLPZ.shape[0]
    gsc_size = pGasPressLPZ.shape[2]
    pCard = np.zeros((gsi_size, gsj_size, gsc_size), dtype=np.int64)

    for gsi, gsj, gsc in gslinea:
        z1 = z_size - 1
        z2 = 1
        x1 = pGasPressLPZ[gsi, gsj, gsc, z1]
        x2 = pGasPressLPZ[gsi, gsj, gsc, z2]

        therm = x1 - x2
        pCard[gsi, gsj, gsc] = therm
    return pCard


@numba.njit(cache=True)
def calc_pCard(gslinea, pCH4FlowZ, z_size):
    """
    pCard_uu(gsi,gsj,gsc)$[gslinea(gsi,gsj,gsc)] = sum[z$[ord(z)=card(z)], pCH4FlowZ   (gsi,gsj,gsc,z)] - sum[z$[ord(z)=2], pCH4FlowZ   (gsi,gsj,gsc,z)];
    """
    gsi_size = pCH4FlowZ.shape[0]
    gsj_size = pCH4FlowZ.shape[0]
    gsc_size = pCH4FlowZ.shape[2]
    pCard = np.zeros((gsi_size, gsj_size, gsc_size), dtype=np.int64)

    for gsi, gsj, gsc in gslinea:
        z1 = z_size - 1
        z2 = 1
        x1 = pCH4FlowZ[gsi, gsj, gsc, z1]
        x2 = pCH4FlowZ[gsi, gsj, gsc, z2]

        therm = x1 - x2
        pCard[gsi, gsj, gsc] = therm
    return pCard
