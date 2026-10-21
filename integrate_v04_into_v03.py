#!/usr/bin/env python3
"""Expansformer additive v0.3 -> v0.4 neural integration (self-contained).

Place this script in the EXISTING Expansformer_v03_Research directory.
Default is no-write dry-run; explicit --apply installs only new files.
Original v0.3 files, datasets, manifests, staged text and checkpoints are untouched.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path, PurePosixPath

BUNDLE = """
c-mb0L$EMRv@D2j+qP}nwr$(CZQHi3Z`-zQ+j##yeLG&9p41>m6&bM-t4cu{7z70X0007jmRU(xL!FtY1rz|F4H^Id=|5DBfkj_ZPDNB)Nl-
;nUXISzq*hJVewz)U_evcmK{$Ct3cK44frWf%gN-
f=eqGL)vf)r^IjT{Filp`Rbl%usz~9JUaxY=g%a}kG4iv^%6fWG}Y&ZLWSZnuX@O`sOj&qZb*003>_cPl+CqXe7nHKhEG*ZfR!?l?AKprD&Mnv
_ZpxtUSZ|0Q6!<G#Ni^l8Zad)G(P9=5w@Txc|FeDyr7fz0<i4~MRqznMz1~@-{RgY^2oS%H;eO;0?=$=ZU-
h0KW`1DzDZ{NKX6!ODz9?zp@AhOm1eBhs>hKq0Yepz#>l_bXogPBK(mN76j)Huq%ovP}*YcqTVLZNtp*iI7Dz)eN_CJ2A_!Vwck;fxgSv0baLm
(p{$libPBiE59}rp)=WluFSV2p&#K+ZCY23D{Bj2|1z-
W(vIDz+ZW65FL=JnrI9gKw67gwT{5DF_~%cv8<5n>Vsb`J225{#ygX>$~Lpf7S+i(y5{RSqApw*HR(9`HocaH5`B=3gbhL$e2XyP#n6hqvSG80
XLw7&;K`#Q&~AAOTE443(!K`OpH5D$zW#TgPH(r**VW_qr`1ENGU?OEcZ#X0W?iT~ljg!({_Ll&1wz63mm50!=qFKl%9&sXti+3~KYtyaAkry3
oT^=VvDjbt)m_wezt6IQB2NqF?iNi{Ez{PBRyL?qal64T2=|=@0~bSi#iE{8Lby{!a;4Zu`NVCt4*A{&_K4|3Q#6)NCCU?>Wc-
O}k*kzS_ppR~I~R-
Ocad^ruHX5oBW2>0@WMGG;Bg&bRb48uFg&dpbGcOu`hLrBf_*OBIbQ^**Q<>5EoN;RS4#kggJZ?CbNQ#51ZeIK8&(`i0q8A23)-
XE`9XmY(BxsX#8^~aQmlRo#Pml3m#7d3OxNnY%gZGVKcBA{tVEQ23tMb~ccXP}g+=YY65^1lZBrFyj%x#8&$+33QU$O=!=T94sESzOoSEV!0#~
4CCE3CZf(<w`<kgN%+ORk84WC)z9#yYpkiVc_VK;#p$*N$(>P_Bu^kw!)9QU)h7!APl;a^#9OCA3AB^s?4&r$Fa({N!y48y){-
WH)k<Gx$F!AC#P6Imv>!qoZP_(+%eQf;bHu6ND+#lnMy$)nnJ%99HcKUM<mkJ^rws%I(pVbC$BmcD{Jp?&H`K(P4OoH)ESouShgPQK2yjs)kv!
ElYhSn*PMF)O?Mw=xFUSa}b9+vP9!h6|dsd+%XaUd-OW>xb+!MHu&vRUjCUY`+mtF_#MQTkXi=Li1PZ0=xLwy|I!~hNPQm6{rPq>0l2@9Fj61;
P7nUhy%@-Rt@-0&;bv(UVuDJ<&Q%wb32aswG~#Qjg<GUuORC@Ca{R8;pmu9Nw^cghfy0fIGtE*%QonapO>{VHy{-
Z(>;Mx)sdx;5w#2Tss$!1JXH_(B8`49fh6Vw*?s94RMhOWA*AczITu@vqQ2v{>|6pON_*6Kaz3K6U6M*&$td_UwdZ2!fVt-H$Q3!NkzD-
^lOBXud!@&>T?NY(1uwB#vxeKut|&M8R=5tfOBq%y*udc^Pg#kN##B<zZiYwZy$xsQsaO=-
;48|_LsH~Yo|VfJAi01<Y`8{>Umtj;fVhfH&yl(EGHyGLfJoc{XIdE6ikf1C4DywT;Mv{oQ5$a5Q8-
#==(Tjo481~uM*2I4=0M5&c`fA;uLm{}vP@hA0aBP>V0kN3(?v+PVL*6%&ylFf&H*v$g5h38R3#Er({6ju=0@CJc#Y2M$LOMV%cLAb?t<J|POr
*`m6e~f-`}u~C{oFIG98hZtqHdP#PywdhFBA4?}AP$1W+?9YgA)t82wUC_no=(zM63vUAj5FKE3hPHI?7y$V&6nlREku#^oJ-dUg8LQ*?E7yL}
y5_49mmSUCHzPPt_MqOQpwki!4c_^juD1KA(|0Mh?yJo102vAwN>p_8e;ldGMxz8eDzor7ns+D!cx8_d76`u(Qh8={A+P1|r3P8iX6YN(p^6G8
-(Ov#q@B=U(5i-o`5l5f<r-
ag$lX_}}KzvK84mi)QBt*SJ*gVdsUBw)1q{b#N}R2^fU99Iqj4gwENLKLR)aD#Y@XLJ|%0ZH4Ijo>CS4((4LzWh12S+xSKdLpg`Ra@P2Q-
YrcU#}mxSLScZV6OW@`vLHB)%vPl)`pQ_9?PrUK-Lq~5+#jrv_h@|u2r>cfV@8H%(<@OlC9+4F~1*pPpiNcCLF2W8%E!vlX%3hvMPwt^oWhLPz
(^}pr6boCxYyvkzEkoO9#-
KLbvmUT#ItX0XC4jfLz!82Ddf~6!O?_Ks_s$E+J$eK57T{ybQF>La0?;bwrc4EWjTw0}}OcijIkaF4VKq)Da6f_=FUu0(ei{B+wkiJNN<U+0)N
vYX`v88d1wsn9VHLqD!)g%^^4T+b7;ImO=t5%p@2;En%Va(gzx$_G1#dUZ?4kqzg%{;qPLu2MGVErUCd_&2%z+3J1|vu*bZ~TO0GaJR&cXQzH?
+UK7S}^m(wd2i0-YBWE?bWEyu0XvBM@MVdF(L?x?Wz8Q`p=~-
5_@{O|ih4p#7F_4A7qsR0a<3r%piLfJ_Mp&gqA2F}`XhMVqzKfoPF=<Lq{?(o#f9$CGo8<LmXo?#@`H#cLYo!qKrnr(38QszAVGE<0M3zNjKi&
;TTc-~E+5yRL_CCGA_XVNQMRuL1V##oGX;v`-
D%(YDcW917SKUb~ZHbPW{YQ?&4Hq3_i=4oENOtq4LqdkfY+8a1MXZIDxG7`SiHDn6Ge&|leI^(xLEX>O_Pslqe4D+k_MF<^Xyrk63I7tGv&_>I
xx2d-
Zo;o(t^!Bw`5kW@wveV7W<q9ABDEu`#f5aT51gdkyBL~n&V$(6GDN{<Nt=2=a<V1d*X_jEWv86QhY)h?Z2M+S9Os<|SyS$9w1!+6POMmH29Gi;
EbyWdDGc?kHkXd$c%8>3B@LdT-y(AtMudBu_iVh<T>_H$vGom*@IJL<6<*5(TvrKy?+M#T>UXbiu+ZI`j43oHcg?hAq2*Q;5)GOA-
N1AnK%*vNr!s@Hk|<yCE802j14;JG#r4EacNVCNMH{MBx?=ZBg$2Ld(&{WN7*9)jaK(Zobv<!w)yq%MvmBm4a+fu})WaHC!KLsuCXlXmuAI>3R
$C%w6vd{@Cic@~$ya-AI};<SAAAi<bqf?B_rZ1i2Ajl~q>J@Usd%3ug@dt4*%X%0vN0s-@wHkJg-2|9ujD?`v7@`JM2tgnv`UhPOcB-MnriH<Y
T-
l$eFA+%7cdz6CO*MCJ0I`a7Y1R1*#z)eY2<{#%5j$Pf8$lh>9D*kX)ZhSVfK^r`nb;ljl<J%`9!mM3HbD$gR!z9kcQ*!2n}Utfe6<M5x0a0_$F
~pTl}++{M&toPx6lBGj_{PnhG0kUl$n%=q{rr?NTj6Z${<Ve;QkFXZHmXM^8l-Rh#8MGKZWgO-8%<Qo-gZaAPJUWk%)AYxYexuekqP%fS0}Ggt
r9u^0>h0HXg;6MJK4dO;;&2}u=EVHH&+QGF3nVM*oxqO0=%NSm#Ruf~)UHaptR6ScyLs7+Xj=|rYSO)AkuWEIJl@y14y0lUQDrJV}^3u#%)xVQ
=9llMmw=NZsW2~T`giHLhK{q51VGyJ`bE?-L*B=wo~(aNjEtJcrUSpKAEcUx>-Su1t-
Bt`6Aw*jzUN)==7T56Lli*<dmYb=$lA+TSJ*j!%a)?llcT2?Ksta8S}IXSLOWdoC43|73ln{}A13VKX_+peQ)rO<&b_K-)-
W)<{zzn3YVO+=e!^+@5+)6K%fcQT~dmr=6<!@~-
SV;P@4Tur@=>3#?McDV|dfqz{)qBSX!z1rx~9D`zPG}NzljYxpp_lqvFt?lTM3Fe$DVq><eo@l*o@G|Z3*tXBnPR$N28I&eVs+P}}6W!cWl7-
Ol>-6S6rwm+a;oq)^o!TPovQX8DtZ9Pw$YzCsR%{eTNi{1T-
Ov)$L5f%_mz^2S9JA}~<tnQOyY6k7=pX!XC1v6P0~T+U`)YVaSDTZT)FGEX#!CfX=5C&?Q1th~2{DXU{GP`G3I!(MLq^Y)V<Db+{L;nq#NWD^V
OFfz?o_AVmghl_A=CZCVGs1eO8y&UKOUH+R!(YHtiW8T<<aRj>q2)GQ(pZpG@ZT7&G)SvdQs)aGcnM=wQg(ykKEN6<w_l9{Y!~$o~r7uT9~C!8
Xy!1rv;W~Y`{MuRWEa&!QUJHy1?uUC3JU)-
n3q`OLTYpw$VTKt)oWks$41sQh6C|8^y~C<;tz&oKINo%?@GKyH<MOUoaI=(^{Eku>%nI&b!63Jy6rjQc6uG(CZ?yU(!Wdf7h6ZP)~9jh>JetK
y%zcloVe{3R8>4?yxO2HF|21=Zf8IZw~r{{lv@)goUe6qCqQUUca!(MtSnf-BH+>DAaWw<RBRMX9X=I>3Lr$c_$!e1QEn<DY@x>(as4;S_&^q<
h0CY-
d9kLP;K@!jtuS=rW)W^I9b<i;Pb!zW1V&{%gj6MRSMpoiz*9dH9CBmg#n!Wy)3I2z%;8bZXWKfr!Y;3$Ab&uoHC}~9_DWEGwD{rQVc1?Fd!6)`
=nk;JxW``9?Vatkk}^(OvTH6tg-@ev2!(QXfq?p*~v#stI?ydZgF>Wh07uUz9`%X$a?5^_>NFGk%m^1zLUB!Hmq9GW~+Z}h-
qIsu$6U*R;+Zk1}7&|i#04vEVCE*pQ~)KTT(R6B><q8PyU(eI<U}mHWWBrChnPNhuSI#mLEIQuAQ>4U^3t&{MT6X^n&aSDhcJI^$XNpokv1xfS
$m^?-eetqB<{SQ<0OZnRK-<g|&uuO5AH*f>G`-
i>lkE0AW@EjwYwQ#;!<X=~dUn<GR3~dq&AhRZ$GGq4d@%2<Ha9eF^D_^D}j>5Fk&^U2G0~Gco}B)Ct&C5ky$>qyB?iq<tFfQQrd#PrE1Y426H5
YaQMxX0{A;12G5n{J)_ErdqIlkz2z_+D<e@{heM|`*u1vT?}h(dO=-OYfek{R_}~6OIt$hvX@Cy1OKMWn>(dv3{Mw-
*6rj{py$dudDiw=X6P{nFh;8DF3H`Hpo47*5HR4TgIp#=XuBf_RkD3$uHy>`SWI>mgIrvDy)&6b!cdD#EoF%Zi>;s4wxS~N+Gd8dv#yL6@ln7#
rP<0ltZ}j72#A0Ne7?Qz0lNgw;pqZI*QY9|0xm;`^M;Y<c>*_yRn6mDbds7^c?`DCFF&f$eLwdmD1J({nKJX)8JZkbJ!wg(S32(9Q0Zk*1n#DU
^Zp=P647+GP$#!)>YY7J5(dBL$}SLi$beG;AePUu3JvyUk)cTZrPo*$0G3MErxJ1$zLqz&U9bRc{H|*D!FlEwu{e?Ii8g4tbH#(rcYv4v8irJK
v#44PFYZ%lC@V*@&2|8FIW^x#>)095Asg-usO+>Yl}`KIF~+6m7EI>Twn2L^5gsYo2-
LeE#HN$!XKJpnQ^%P{=IR2&Hv|nBs?nEeKjE&D=V~%3DA7g=G6-
Cf0+JX_R?!v&pdDKRqr{8VnW13%g)XAJP&Tcopg1b88q~9p?zP`9XVewdZN9OSP|n5?tk6mf=&EuOj5e(P5#PE%=<b}3CO%jxbQh_oQ?{`TgdP
Y)^do}OD=8Ci0{h;ncQG_CFoHy=fEC?_#Ifa!?Gi?zD8n7kJ5w$Y;5QfszBUxf1sNJg$9B2vk|NI>Jp}=s^=-
b8xD#~o0G?*z{=4hjI5!EqVaPY_Zl}1(WvnehR0{Pr;~aiOl?2S8n$aZBnFIs9+KR~c=AW4K6}-
iTAd^FMkF#HF=60_H@6^3fvWkv%7Tg{1o?Zop;ydnd>DEK+Z5Vqy7J8jn4m?k@*oWP`qB9xrgD#DQLiIIeAuCySXTgf+f2-
RQiY0yE#`~wbrWvhR><CmQQ$)`-D_Lw>6-?XY@m@>pNYLL3!`zZd+uedXCv#YxxbXHn(P*OC3FKF_wIbFWzA_)MFOv=7(<EMpzzh^-
l#CviRerks2GkGc9WwkGDb%!$Lc!qbxf$^Vepv1Vh_7Q+K*ekD2YFU^Ei4DTHDE|^cVPz48$s>>+NWAAXGNlB@EGiZ1P`)giBzlHSQ>{$1zjSL
odcgrh-OwZxl$;@eg7no-K(lDizRO|3o9Mu#e?dx2on4&FkV2{(n-a02r57H<0f98IhUosEzEFYh_ZCH$5_+0hiR5X-
fFRz0C#Lh*`gIUvc_v;YRPA3M4@gF3?L_1L1e!Y_-
;C?{*mBV;QzEh@Dl5i;ZQ*^8a6x`e(1q!vAm1*0aS9L2K5IE<O$THs;W+a{{a@XvtmqzuYuf5TAMqofTMrRwoKDOb#p$z#xqY_$Pi8}tuTj<4g
Ji{oB>vs_fMmTgB0p@C8x>4aN$XLUwenwpsrX<8F48Z*j4wDBIxsU3XF>pXB~M?=CzECtQk<da>_OZC6>Z*;4<9(4#f6HgC>&7iN%^vL8~9Hpc
LD^l7|KgcVK?sGU+Gzr&qYJ`~LY8P}_Y!r*eI247?M&Kf71XOEOpasKUeh;o*9#8-@U-9MQpmtaZ-LogxtlNSW3XI5bk$grrx7kRNNuXq9eXuq
gcF_56jWYv-C8V1UIK#!^4rWr|o?V68L$ffEN-
z%~a41t=wB+$I4XMicJ0k4x<g{~u0WWEt#od;otOk&?tU)wno`va6wo+#et511(IBj2{Od*M6<?ZVdsju;_#;z%P9B5qrH}FXuo+c;@gnxSL!Y
Ie%V{r_tH|Q1a0lztJHte*8cFEL>b%{GS`vQOe1f@FHR9u9ri8+M~OELCbV@aq>Uw^rK?@zgb-&QP8kL)y^-
y^Hb!Wur?OSg7`6P8s=YZbTJDado`S0o(8Q0^65{*xQTNL!+J@eDw#1g==))pXF|#scS(Cne~+yMF`I?_o2i<Gg>@d5iO$;KxzZDv1InIEr^dP
o%%&WU*>-f+xj))&Pz4CdJNJbi5^)w#V*UEB*$JEx=fgg3&l>!792wSg6xUWFoL&|lu;YPo@WjJqLUj4$Ua!QSz{H?R4-
Dy>N9S^1$Yf1M`&?S!1Mr_!r{gln#8ReuM#udPd^yj<>XT0}G4-AmTo1)nsA6NCVj2bB-
?$hZQf3djUkK2tWRg%q=GgT9^d#Ku_8$e+u2mab2)#r2a)Vid`Nao@iPxM&du$>9>dAc$bc$8gwWPRYx}odicfy+otN(IiX+?fsCSg!5nKyK`I
VI;dScxD&zTe5oaUik@=M3j^f=ikhDOvKC(Lzs>bGDe&eEH?~rxa;vG%>D<Emj%J52i?ZU6&+D%~z$412O2CKHY4JW!q{u*$NIg$yzPDz^2I0=
f$<yYb5s76lDU=S9I9VG(v=Uwmqeb6=g>cvt#{HWlv`DFirZVx+T^Pw||F2ir_qa#ev2`5qK9H%%FEcou9%~Aw}*6vfXQ~dzHs=6_WI+A|nJH@
#J<GY+#<i3X-VI&OgBa-3nAV3V+HW0RYHQ|DP@P|Fi-nQ9%(|(f`{5N;S6Zj>Zvv*VIwWf_5+mN=Eafk+1~~JbL4TakuVCpBf-
a&^C~~Wt%#8WGA-
63Gfr?r_+yBSScs)<lWGqVR(qTHFejN%RfnT`B<u6p|7ZJEMCZb#s7ZZ{3ZBVHZ!Hx1XE4LyXaJMFLKak)gtF|pfm+btI}(=tg3A4rWVfI81nA
r7d)<mtDIFgwU$)ft#RA9Z(0xWn|yiXE_f8|TFW=Vta{0=EN)bU*-
#Y7nx{5Fnir3`@m)?%fR)WnvyX_c2f)oNtyKBig5Vhi%?n+<;%ztc_vZlrJf&lsrOT--G|y04;QivWnnHV8Lw5%#A#&0ac0z`iRjyVK0(OC~5&
7|y;qGd@`~?(iSODABS`2*Itp+I4_lHv_8b7L_Lr-W+Un#q-
&C7kj$FS0v@u<s3sOxHh2L;NkYL1eX2hB$l?On{!$bz$a7MsdhWCFcwoEvgseb2|UNr%O)oqOjsf>&*GL5&*?gv5Q&SpFQ{e0(3<pSG5gx4N&E
F5Jt>&)Cn>(b+9i=YBsfqpj<eyB4tY_49Xg9_ub9PDXBC9wuzR?>pD(ArpyG8`svWLArr?szhe+&F}vi{d~K;obIr!)BQ|rVGZW#V+@C)#qo4@
6&qr2u(8-mTnrB4Y8gR4t!>gk$8qbdw7lc%>!+1gp`7vdaFS@oQS|yq?pcpi9*Y-
)6m}l+xIz39J`yO`juldYkbuc;7h8L~M*YLLr%Q<Oc4=la=ikOnJl+-e#-
fgf$8TX|m8o~JQ<<+ZwIg9$w+P}d(vpPaq2l>iUY6ESIKS>|Oc*#>Q5$!+*9K3gYpBybYFIw&>RoI72!)R37XoWyMfaL<n@%cOOm)>gl#Yg{Ga
L7y1D4;4LXhRuymMvOtZWh;oxD}3Ag~-Je8Cet(5!E{as!!qia9Mk8RCrPuFENC6J4cHDH|npnDdDF4mbNG#DhW!0x01*qc{4={AG0`Gcr-KNd
3aBm8I|yZG|smiE=f_pOrZ2B;RO>4AMIE$G?ZEZYv6Kbk_YWyi`zjW@T4m3xHE5G}EI&cQs^&%0=F!6Ooh~@58g^D5KYJo2ZmUnV9t-
1dC(pXC@<O$#S35NF6@eP(E^h;8N_g(#Xlk4gtzkTxZIc%!HxHS;wM#0p+n?gP<)CWI0pGw2Y&($%ZTH*j)rWfKH>T<(UL@SnU7D=`ijrX{V!c
&Lpqkc{rNk79V^PEF3px)|}k0=lpGqI>Kra-fRkKZgVrV1@#qkX6vDwHSK^sp`~{s0f&JY{f0v|FbccPyzJZcVaSp-W9EDkt(dm-
kF3A+$v;P~zK;G6E~Y#4`W1N~3wmOHLAq<T_N>5_f?26lSfR@VKt*p~O_<z$54Si-ZFEDcMa)Do@zY^B-
8_qr_sGC&C)_Nl?*78W?lSVQM}R;~_8K>dS@}nVTO`Z}RJ>WSc)^j9bjVRSg1G|1L@^~*Yh}0y-
V*Jkx3naQokV<*eGlgTp%bcSJtIu!Za&2M;%v2J9cAf0prr&^n?9zx2-
pblwTc|m7kdlC2}lhuF<^<>Q=*woz4Af1RjJIR#q;I!r9HvDw8^^+1BdJNR?9Ppq2N1@MdCY6V%#(Ig;4AV&W1>$H1mCDE7)Wl@zdm1b|?VaJ3
!wTJ3<!3&*aEW$v)e@c)`O583hJm$b{VK-dx3#$^NMnj@vWJC?1Q-
Ai@|r%qhHo5@Z*7!N9N+#1nL97aGxMg}2~7aZ|YcQR~c`+kq~V7vhbgqKB512yn0|rU(F`Vl4BT2Gy#62;dnpEFdM$kUkhB4&^9hY1vB_?(1QR
p&dZ=oiuGlar(Y5vIuz9=0xWW76SCV)Mv>yGFQj%+r#rA9bGc1y5QIUR-C%rdN?A9L?~2&;+c4~oMc0b?+)l+MY58s1=h119#O-Sz3z-_-
f>w`@EiX9w4x~wt!u9cz#IH;NV>CKWn{R(p?Ja&k08GFLn1{1w*1%KUIRygk&SV}m~|AR+ZkqHq5Jnra@-
RT=S1oNr7Og6O>fCiSc4Q7r0}{A#@1hbdT-4<eo?sOq0B`HvL(@(%$JNAEZ?XYP5+*|&WHk`<BzU5pkrd;()z^(WcvD_K)fUXu1)KPS-nC`?(7
t!1JxCnoXO6e5+fDE4#XD}?S-=vBbG}<ML{`@H^&d>KmM!2dW6UIV*LifL0)5BJG90A-~wS1A605*tcEdL?<SOeogrH-
E%(_1L$vpTtz5?jGLkjq`x+(Ie@dKUL?%*F&E_ciusi)rI8WNcObwwzwIMNB@CHUKMp&T}iIV9kYk3q`&p`cWl}rIE{RCjrK*Dcf<@{wW5&LKm
Pz5k%RQKNn@R_@fs=iTgbSSg!q!~^W*$ZI4Bm|u!o_4E=bGEPKy-EfSi~6%JGp@*<oRzS#K7*ER2=CRE-
8dvbWo<d<(1H2W3vf$h2FwMGgYoF>nFoj=;zBGw_?_bjN$5FSmw*>NaU0hcz{FT(mtpdzTyJDQ#?$2w?;Ym8KtkK7T<_kE22xR&O0A|6f=70=T
0>D^k4Rp3ggkJyqOF~YZ8%%>>qBfGJnLE<-_w-(M%n*XQ2(`d7Mz{B-7J`FpaRQ=5&wSkc>(QUk}$s#7?2@{;1=O#Qng>zlk5{=8bg-
A9f&2wLHsTnfcQ8zsl@@KM+)Zm!|-
q~jWX=TyPvztvl3|9$)SwVoCkUsl|2~39;&)HjT9~1`%=GwVK{nnsM$8q@Sag_4ny^K4|1ib&n}p}d|GNj?JFAw&{3nrM1?zG{7y~;V~M2Q%S@
pTL^|oggMv|iB>VthR5Mi@N0`^FV_z<(PgC#PQe3mHEtC05UGc&P_QkgIHzrQnDuE>S-Fw<D@eX@CU_gNA`+7jDv7Q}-RAS-
;O3IF|0SYqD%*LZ@erHY*mnveM!@Td8)V8lOayjKeGE;Vi(bx50mZ=|XPco1LJ(xlbAs!GWP@==Px#r{u6n5H=V3!a4qk6srMe#fZXcpf-
^=TaqfdSA5YXI#Fp(cM!z|O}+(VT%PT9(Gl#w}CoUHzZ~j+Jp@R(LD%XYQ+ujb=wq)6_NIm=hj#S~M>Rb|!};?yr@S09H4lZaZfVuPIWx#fvV|
Gm+3PIH_hg|KOV@9g5wIOX9m;z*HiMW)|&PJRmYjiA9WGV~dP{3Bw%T*w}eXKG?K*BXf}OlzZ0_vyBh7mSeRq$kP}i4iG<s?#&|=%8mO4|KG9B
dTU105f}g<1PlNG_x~U3ltur~!dG8dUQR_xP*~;vCOp#=?Rac<n9#i+YKZ3<UHK)mEua}?tP*sTUIN>vtWwg=;ySUcs3iAX{XOAWdqI63JZ&O)
e!NF!uP+|wMP0gN6s{hO+sX(iboQjFN;EguR;_w}`;`;PKJEGo^u$b|s`oShcKt;)F}@I;h-
C8Dyr9!@)o)xv4T+vup=6m(9Cj?{o3kXIHQAZ?B#AuXotQzq@a=KR^AZ*Dx?>K)Po$B-
PQ~q`oWkMVg054eJsN_R*%<aiUO8(~kis;YeY}5mnxsUN2Bkk6ImoKiYvEW<N<uW}FkG9Of}TLS%_$6$RR#^8P6cmjHz~a$UQn(ik`TAUaugp0
Y7|lXP`O<RqRUZ0B$jlWuA@Eh9DSlQ<4ps}PDSkg-
@G+_Sm$Gdd<1w=M|h%eN^u1*E<=k_GZS*Wv7_~WP%wm@>pl|RpJNyc^&hmpyAM=((%|<eV7-
cKhA&P9drUW4wxV)>A_wBT&_+ejvYbkGu6C&9ci>*R({M^s*$Yrqw3_UjA$Q$L<=J9{ygJD7=rpuDyCOU0MM$s3p?_Wl_ztCnq@T^udi@xAL3R
xSySR4w$h1bWPV-nfNH!hAPepYho$ha_3{EyS^eCRxiiPe#^43l;bQy!#betnc4WDKO|6sZhRVz{Ddx>Xi1?F`g0gq*~TU+;_@d+JFTDcBHgLW
IB)x}-KbQc>w-3spH^K`zW<M$8Hc%w&n$6G%YF)b)kR^^j-
XvKK4ae(`i1L($E?W7zR!i);?%jrC{dS2|I=Yxu_!!TPwz>?+4uO0D)2<qt4a<ys<!yZ(Ig9RpAk%Q`6-
HuuD4Q*EVbnl#EwB6#{I6s3p5U(b`Vy86vw8}z=tAt5+2=v@yLec}V&SzF++uS?!Qor&Zc*TXm>(ZWZx8E7?uOgg2=Y0~gb6*bb5B)OYOOOfpV
DfYLg8%yS7RrKi;P^2x`wp>4KVeS+Gp4qmxAp2&YkC`|V9p0LAqLGrt4yH`J9<edb*ZzITD^f*z^uO}t~|xBsxjBbu}56*^we&Q6jFY>1U(^NE
K>iL$!w;1w1Kcg!dS<h?Rj{TnzL9&15mVypqBg}LV8KEhd>E%G4wPf-e5D9qbkZd&!dC1XU>Hj)J-
+`mY5{UABM<l?mRZ?B2<r&j9>8o_pBaqJ?}9C0stHW0RUkCzq4v+WMk-JY5)H`GwZw9d(v4s+uNlk@yZV|AcVdBMk$}RJc@rKVjxm8^y75iENI
E*7<8)&ul&U=35)~)NwkG`pOu{Pey{+Kubu;`d9*N^uscYVs$9pg!-_o%$1>4jOpMJ|yO!0#U98$-bQuk>lx86-
IjAy?43l>bBm9Ag+1FI#;2EQC;dWU?oBgngqLC<`0nT9AXzj^&JC$|hIi1uf(D3Rg>d9364*#sVSjcBs4VKGbN?I3{mRZ^=OaDZEji^{nh*U<F
5aH_TjEwC$K&)gK#E4=ff=YJ@S#o;#&v}^ETb>O;lqw-llF&k~2%%guA*<PjLrzcxw?(%#-
klSQTnoYBakC(<A#CB@?v%9|LNkhvAKYG$hMVFe(piR8YI}J2fC0BUX7_IFf7{wyVS8&7Y1?8Sxo=#5|9|T4@g&jE#eZBYfB*mx{=ZzinL1gTS
sMQj-~U6mv%Ra6vFZOXo|`Odzr_F(cK3lAQtu4x`!2j8C){q`J>-
Q2Cft~*Q9<jOltV`SxktWHxd0@}h0*|<;UmdQPsmpXfE0}@6+U4u0cVmMC>p-
FYMV4#BYsxB8M6rvc&2QVFl&hX+l%WM)w0M2zbGerX~>hFwN5Iso|VM<(L!Ri1iwgS?lR67CK16}fve`BvP(N~K|$^PY+;<jl*gRJcjhKFRmT<
5&P;du#EP5(B_}aD1xu~jXVJbe`X||&Dy5DU-
}iWm=ekm4lXn#qFZsqYm5k?YEPv<dtmfpHncZAtaxf%4*&MxYPt7lcONw3zdXQd;2{}}ZEjYwL5C3I*!o3_yq0xs}QsC(!@;W)}V+>c9wKL5IN
h~@F<e?Wu#N~10p-8`5Z-
iJHN|1u;B3v>4RKug*xe2qBTLv?|uUh|G1bXJ^V7Z<QSULvQ!#UvIdzWH808UM*^VcD5==}(hT&?i}YG>yk6e4%|L-
FSEVxE%wNWNZFx23oFW#amZ#<u8LZB-
u=7xB+50~^?tTov97yw|b;cm;kHr(vui$NccB+g@J~8{W|#SUcK~a;Y?M6dZ#_&RE?GzO`}aTo1yn=@pR1uFd`+K<!4Qq2}4=82ZF^Z9)$S58T
+-<geH3^Z(<YZbb~~z5jWqC?Ws=#($`(o1u-
Xp^K@$p^2e`i=~_C|NZ?DtFCRo#fIYdR=?i`K9Nf^)(exN4ok6mV`&*qxFwB%5F@goWvWD6snd1y?~c2KVnQy96=E}jhdpn~efHzC+SPvMvrEy
s(oVzu?p3Q-hsXQ%+E$-6ZlYT*>y#iBnbx|LK}xMADR+yRT3fSAj5=&DSH-v3#Cz_wsaCDay=J-
W%C=7<=RYQiX_X&mYjb`nP@TK9Z8IpQb|W71AXOW??6*`Ab}%(r1>4p!DJA_!cl|pwG&~U|SxKGC+`nQ{Yc1fWUv<l2C8QSE*2Fk4HZXl@>$2M
%-!PZLG!>=Y^Ou&;S{J}aJ$Ja*M>wtFhT*7F+=7-jMR}OI<uIs^-
tyI|`*$JR!<)vxsQJ(nCoPq4fi>@e{@rG`C6Os%Q&Q<&G?k}s#B9L1&BBAVbt!UNaxGDy4|5V@jUb{H+SJYnV);3@MH=yMS3x`3KS^fAv|>rYb
tTcZcd?iH2n|<L4Bn1Juk8srQX8Tf!N>b0ZF6P#jBD6GxUWRQAchY-mQf16J%#Vh?&;B>MN6&9C>)lY=%wLv!w$9YvSU_<AV1ulVLE%44C}S8q
^&e8HwNXN?Dfu~*?f1}s+8K_qzOXR>GzGV$69=lMEnfxT9Z18{|Sb+hf_NyXoDp?CZ_1-
)3obZDT^mw!)_@k5$s;W;1@Gq%MP$@Er4}QO1*|#cz+}_Zk9-
*+zU58XnIe^Dib5_{rd!y%%SP608f;d*mXEFKD;x6n7ud>ue~&5SIilW#&7INuP6>A$L^z>9>1UJL=Oo_!1r}KHCdvO!sL#KYwvpg(D!IJy;fu
E?rqkoe(mdE_%UF{k!*507RX9~I!Ge@s=`1b^h&d!Lw*PAP`vfLn31ldLk0r64Gg$bgk<aVnmqXjYOY1LKBM^?B`X%LBi6XbCK0w`1(czhfIZ#
1$cd6PvK1`(K+yU4R>9rHO*fgcNQ3Q3kNkN``zWoMsS_KA!R+KCNWp-_b!?4*4=0x*a{H%DqD}p$J_H$65@>*AL(24M<tH@FijZN%=e;ESEQzm
tA&kevMeNI!ad+Rr+VCMpRVWo{0Og>z2iyhe3+y#5Oyj0tgIk@|l?HZaq+)5oAnbr>2SgopGoAw69O*QJg#N%lelB+iZO)Jmhh$cdndwpB{V7R
&3<hoEj}g<F90@Gw2(nNJR{WU|g*PC;K48ry+$x~5%B?VMK;-
+&gMlYaGY=X;fj1P4!qh~(j$z7Ev3A1b?$uzFDJ{Z}%WNt|dVr&I`xJrNuX~9!aoo1fp=7^2&36a#jLu>!C(943_TK%TFz-t-
B#)(prOcR}gX1fQEIj}hD+NA1tFbAS3O0LU)vQy05Q;PB5+dHQ)%o+zBqc_FTsi*_JW-
`<@20$S44X+19##^GR6aPie)Iq!BhqC6a)x(l&1P2+g+U(m8i~}K?Uzo-9Ap<1M?LEfX3>IkBa#3<7A_ZAt+-
$5F9lQ9p_*YC0SvtHjOGUj{ve}v%>v#1g6zr2u?gf!c#+uF9fhC%W089I)Bm66xU1_$_s6SMnwxl1*(+coh@5j{JS2}9{o8-
{4P|6A&S@GpFRBnZ%fgOi>#z>Uci&jwh$mGT7poJP4uxg}L)R?upuPnJB(LfoI5ZeK(QMn)Il*9>$euzbO-
yUymGn$9={)DEtXlTRRSrw}BpZiqe#L(Ebe2zMby{SO(!{{?I!h_`OEQ-EGzN7=ACDQ%saxNJT)_~JeJCtSwxw8r;lL*rHGjCfmXJQw=YQ$8br
|D{;2`6bD12|!7lgNvm(S_d5r_V6CO+9NCod8QTn}XPVS^g6jX=re_*KO29V8E#KDw_6Sh&f#4~c*5=how$^oZs~sDSS}c4xiO5oizYM=~EN6c
*2`_eR3a2^-JpU21^@VrT3kLbnDf23ImZX7FfALL-Pcmy$A^jKCJ~+X-
n*am&(fSwP!V!F=tCm)dJ<+FY35VH+AvC;v0T28?pvV{1HAls>!%3&Y@RA+4(K*gLVGnU$CQx0sHJr2{Te(ScF7XwsTx7DIwv{=H%Hoe*5sNna
|;4*i@@oY8~>d)Jp7s-FunX|>Q^*AMi<xvIegT#WxgyfuZM-?}?Qp8=rpK7^KT$Sg70H`t7{z%5H`nu4-
QOE1&kt9#okO=SCAiBE2cF;)FLzrk`H-7;HBl>gm5z3w1}r#Xu@Ds-$IIu;8T-
Z+DQaycQN(;hA8_N}KhmCyIn^W#t%l2nKvilbWR0Z8);|GzDdM!D_D1R4M!k_-
R<^Z!_$hl8P=vzfh<t*O)hq#|2;6H}Z2Ja0zJ)_%JK@we|^&`V*&u?4Dq-Os%2YRAwDWe}NMVn;Yo7@-
5{Z5@K7ib!4W?Uer=Gs#O`D{cv=OCpDLVh$#pd7fw3<ix~8xYFEG&_z@^sJ?j+s*(Of%may}D;4y}qEM<ODN<XXd6QIoMtWl}66%y`(_9FYd9(
C{n`Kf&P-HL~>P&=ZT4g|LDnnD_uUyGi?kKaSiAR6RIGZzb2AVcctp-iC)NmJqR3pccCESKO_ki8pAijQ0Clbxfl+1C_d`^sU>Ubj#n^3D<VbV
^i*Vrl&Ka}GbThNs+DT=g`Xt(s>#(#ITQJ%;VT`-
dmRuoY(EybjVvIMQp%uRNfF^o_`I3Qz<H@g*_eX&#fD>6mByViPY)l)^u4L4L=o@4(aA`3ls9NFb)JyduUvNcJKrt0GTVhk!5RjdD?NZChtil$
+s#8LnAQ)P|b?S)@2Pg0lC!I4t5JZVX8E)IMarDad%Y0M%beN&||W7gfFH;2J*?lB+n{v63-
Kn~YPIYm4?P`YGy(kDSzC8h56BH8Fp3BAuJ@MQa{bnW<<ppy;QbgaLF-m97trK|puFjcB1JydT-<-
UrG&<>34OiVW`5!e^g@<S7CLFO6gYE8im+q;*EtfWkgJtN&R@D>y8$jFS0TyN0MD{av6Ksf=Ndhe~MVVsrttl)(kdm1up$~|RAUW}bcYUn0@&B
|45iUFV9IJ`c=*Lrv!h$Oc4JK5070Dj1ONjvr~-
O%cPO<3jQuG;K?sX{N=#J)NeWn%o>w%ztGQgl<bUH*W&l(@f2O&+>+>&GZ>M3TXnmija!jyxc&VSJUz$AYEDW4;_;+~ny$ebH>_$aL6b$kKYlV
L!XZo}nxSGTMW>A})W;h>z2j++{BI@t0^$S*pZQB3VSP$;LJe`H-
|Sn8{R(r1wriV9RG0Iukorj&&;k%j*5s&v5MXl2p(zE=Y#cHiI*&yR>?TOnlu9DN{!&9AQVFsvyQEbd57|o0AJ{<q<EHK2nZYst{w7(r=uCSC4
6ijeh_+r>F22$)tkZ!odqgW257^4CNKgm}V{Mx1h}wgH%D7V=};1RU+nM5Dzuf9^LPy_f{MZ%G@b-N}A4Zo-Up0BMB)a>Z|ogN(zlsVEnOeGbj
pUSwYR5P%8KQl;@DJROAX&?^bDaB|Dm{xL2XiK8z&N)Eo`K6#@rYs(YibH6^u23}c|H@m&KjMG?%7BLYv6%)(BA(SY9LeTUJA-bGf}jIbu23P}
}BOV!=R|D?G^6?5qKZ!U)rY?CuamOS9JA7aYS1)`kOgEQGv`u5F+z{zEVN7@a<ewZB6X(lkZN~unz6b=@3s-
NopjZ)=rkmt0()HYTngvO~g(G~$sqM16z{c$1GOlC#)LqXs;#kmNX*zjFvACk)0{xQop&8xNw#Gju<_p^pqo3{)FeM~TC1F4}Fs>W9=NLJm`Pm
ZpkCR1Z|4Ac2{q<T0w-
Lz7AWUM}puGz`ZoH%lN);nN|UU40_?%a02amW3@3P2v%Lz7cwb_}iU8n@kCd~<8{uRh0(l%4AgeC`VgEAQarx}44C%KRX8P7i|&tBO9=O-
VTP`sMV#BQg@TZ$|6L1Bg^kx5BiSY~iicB4vFZ57wRNlAhxS<WMC`p(<NIX<zD7PA#9(Cm~b7sNk}1J+ekbf~A4|g0iC(w%YpWB3t3fDBLz~n&
-wWNIct0aT{p_kNTCY(<%DgUO?=Os;&bJMc>8ak$NeZAOVV0uO}UyzQQN)<AF96)fNi}{D^;zbXEnI3`Jr2R0C(I07G{<YyiT*u3Y26D%uzf<&
9xo%)UxNBcT~!Euk0GPFjL_Ub`Ryq3X)2-aRB{8DfOUfUyCC=(+c%x}Sb6^}+@Zbbp=U!ZbY;+@)vKb_(yy?9XiY2vIC-wtYc9H5h83N>f7ZIi
j0CXFJKjT2Mnon2l%feLeF`1ZKZ+6?en3VRYt7p4cW1=dw0~UG*zhlt@3Cgy~!hd*`HIkp3(G9@EB!WW^xruRtYWA%#)~;7xPOgf$ztk@~vBkn
Ik^LQh+Iv|I08zgyG<K01Puc(1Lv*wlHI7F>Sg!~)o?{D<~ESXMZDaUp@vT&dCz@O;c!o2v+^|8kL7)<F(ygQ`VqfrW~#sRi2!-
YC<QXf=(!8><HgamQ{uP#JTE7l_YFvb;(<8F_TH&Q*5^jIH%o6M3=lKGLY#eVqd&#A-}>fpsl5PrG>=^AKMaY;$1D;-
fq(#|+6)0r1#kAP0ine;%5pE!laToEMx{#(!<Lvb$F>fY)qwYp?(g$c327Sf6Mzk21VVXDgUtLck8sHAnX)l}1#bHIj2V6&7a?vT0$=)=7iCKm
~ZzxUs0XLH`HR8R`5UF4qe#^##K>p6%RC1=$97NXy0oO<5ts2O%O$C+gIDM`&j?zToha%)vkVWz@39o!|v@WM9Bc*FYj!c;5?>AcW$TqU9x3(j
u?=sx;B}D1OJ1%p*Y{`!aYu5v97=f>S7Gv^E!&1Pd*Wq@0R2rOa%UnJgq*ma)V3Nj9g8$4A5Y>Lwwbl&eq8qT>{^lJUrxISKglfD#|M3Xi#;=g
x2-M3OS_h>ewInPS{2Sm4z_rUBtYZLQ$|OprtDUQS?T%L{oF6;H&1!_nYPuR0WpRS+)2YF=^IuwGx=_56r!4-
pP6$Ih?=z*k~8@ce%L(RdAEN)nsKs|HK{&u-
*2dLom*fjaOvFUt~R;KB?Mx4jHJ4VnuGB1fdxpw6+c=Y0&d`}2M+dTKvHg=<|4fpM&sLI#QS6)1NW?|6Qj3jtBQTu#JQO3hmrsg%k|{b3lMOFS
vya#Uovwl(jHYQ4%aP`{m|xj1G_L4`U|R)29NapnAJVf8w+;eml$2yxmlq^INU`(z;klTZsg-
};^pp+YZ1?3r)iW;dHx>Xb{no9%Ra`4*z==gQ30+sH)iCQYsi6C;P-
&Y~F2E}rC{p!<~yjm@weIg9KCse@*rW}}ElMg#H$m(=n^jknY*9CC6SC#e3I$Bko@xVB?0i_t|hI}`x^bSC=lnxb|FdtEH@C})}(F6#0vY@F)0
@fxaS;42%J<J{dWl{NL~=Z~O?A_K*D%zb$r)0iZ_2Dbxzq=l{m`@kr>IPSxa?>zfa;f}$brrq3X-
lcIIMjO=gq@{)l_rwN>tT(rb(590Y`sgv}Q-
7Fo+7|Q}+!1?;5dqMs8)(AK(v<G9vKnui$BQEmOR4Nl<?~hMd@M)yDuyTD#$|YH`DHpuIqn%Req~mQpr$5sgP7D!FZm?L<K$_J+mQ&F0@BYFCO
}pnpSIDLEGo=;vl%*uWnVjToWXiCUGlKho3;A=@`FKHb93eges&&z=i53{Q92|XH>J7fpfq@7nF(Bzo5w+wy9$L#n#*3%??rrW4j(7y2ML8JhP
$i1@0{G9-^bO+SiAkM-uK7yZd2=sYVV~E7pmruQz+lRYY;6c;pB7N*hQO9Tf}REy3>fv1*_W~r`<lz2zMo_AXJ;WQz;71SFBzI_aZ37Rhd*E6`
14d&hM?;%AF;lkR4t*x8(T#<%;ZH7x$O>^^FY8&!|)vQZ6sRaPphw&P=WkrdHlc@?K6`^EG?&-iPc(GAf!tj;BaIeyXmx2Dv%7xx;8OB-
|x`5rtQ+;6#l!4+YAMQG54VM&EP=>}Om%6tQeudi1Qsz4NGp$e-Uo<@`T`VBWMHHUL{&+4fEu-
94X@u}(d(b4>fJz6p3`S||G^=9>x*eb;_xq}P9qq4@>o$BAb)h%VO{tM!d_T_X@=x|r(}AN2n_n%`w5e;Ppo02JT>03iN{x|lk<IMe^1^Z%})R
y6hPk2(<j-^vkyP?ByfYI9AT1)YGPu5IC=pWV<e^z>r_3P>kcHIZ6LPB>^kcesU>;z*|CoGzq+Nf0S#<A*-
x7B)6E%BfCR!}(KRlUU{1CZ|H4seboQR-
OaTs#{;%n$H)FEwiq4R4!M)?qA&HRYr_m+}zkErnDq(EcI43_3quaRXSI7E+olBFIHsgqkS$q)AbE4^ht^qeagA-NaUR-kY!WlM32s$7y5p~(>
z;0pG}+gG{%PmEhiqM9(8A^K+7pQOSU$8N@+l(Eh&}s6iBf$?~z5)X%#2fa#m%i=puEcdQj^-OtO&c6?3Xu%+M?Y91Yp66K^{XVKjco-lBhhu{
Y%X5meLa#RBQR1oAzJH4WF0*)Idzl4t|%<xG_;ZzH#Rwk)Tjz~Y27C#`brJ>5yI1&)#@cGBthde+Zr3Q$~SEHNbOL>f;7F$lFvSB(7lIT0KST3
`_NbN9GYG>^gyPm0`o_8j(07&A+Fo-
%YQ2~4vP8)k?^WxoY@^lCLz?^%AasZq{_EEUJ{MNRQMYY(IE%?=0TmCE{m2iZ5DBG2T@Ax@9DEFFZIAxBS7uj$-
Y4LUuaaFBv8{R~7|Y1fI<8^-y=^>Z|w(+=MkV>~;LFIZ-zF?4mc;Bt24q4jE?vGk_vzX|V?E=+h;KmNVrd68;znMy7atZo+tLzTmOrbdO6qfL|
SIel2H!6u+?J>FVpco{U=T)uYLF;!+)>Pfp}LFYc&<`IbBKLTU~&Z%RN?(=em6&jer{)wh#CZb{M4P=tUq{U}_y@3(W{WoYE!9|yTB0I9>r1YU
aIaNXy$0wTxofE&38qHJDIS1c#V<Tdc6gJJR%CmmzE!hjW4<jk2Se&h!=&lwMBx|uXT>}H3fFZ!fH33LmN3tO`*w%fscn^$J><A}8^D+kgGlbh
-`@__DI^puy7+vm}`B5Ri+<)>V?wjz%;F-
ZwML!^dr~au%Q`9LrV^t`ys5o9P5dr(oI;W&y?j^6^bzMAvM{ME!@GhM}#J=_fw`IFY!2AHIBsfGut&bV$C^YneT;GO%vH=1m4ESHMvt?_1cqn
KTD`7+spHvn@{vku~GLG&Q|FTL+*Q^t@ECXUt*8Z*G--{-oR5|+~ZxMUQR+p8`A_Y6IH<8LVsTO!7x<=$UaXFH#M}^Q*dMWw%u@cXIRJiuN`-
o9l`Eu}uiyIR>kTUv757-
}7OD9Pw3Wr3P=gvF?pwMV4C}uBP6!Gmbj_Xm#IS8TLB?x*~C8@#&b4++*eLpLA_(M06<ISLgkqrWem*lesG58R29KU#w7`qg9Uz9@Iqy<EEEaU
$J=9oVR$cRlD1xY%M3_AfaH%$M%Muyt_1e0}^@LXb^nKj!p83OjEvU|l<qGNkA=^&mCHF;{ep=*P1uxQZ5orc&HWFUWfdLWLw!MYwCnZ7p7A1t
I?cMvgz*D}UTE>N6j1<{5g3xpbY&oyqt*$j~ktXGQNng$tu&*~_I41MT9PnZ9_E+%DNU>XfZj!9b<kv@arz~-
Z#ed^<YE<_1jln1Ke)HhLL7>0|g55nExCI8Yf6)5lw06EI~L=6HOJ|_r16b3hI59Q|w*q8%P&UopJPNUfnIF%0W3thD+gq~uCTDN4|ycZXRD{q
WAw30auw^r}J#u$AALqXpN`r=qrU^EDgJN>Ny_EpLc9UY{oAa410zdopz<fb{C2Gx{hCG`$X72t2;Hheg+31~2dDyvbfk$5%sCk-
6mV0XDLq`e0o_Jtmntlc4DNF;xxn}R~|cE379V>iAs<BRhg#@`432XyRxAFflci%}mEi7|9OUc<dX;c`X0GhcXUV()p_W3PXNZUmYH@E2|744s
Ilc}TDoI_E)64aL!L2pF8aqqCD-
`Zd==ambn9N8aS?#ddu=fCQeD2XeJ#O8#9mv;Z{|mc=C&2)WU8=5JG!UT*V9tLCx1S?4u?hAJii(yE8tqi$Z;5#6hN$)C4JyI6^eMxkV_<~#(?
<R;YEoPR{#=NHF4KYGeNb*E9+0reXh_)?rA1jn<#9(lUpV*w=%@u!etKv`YUiSWelQB=r)Re=x8zw9_U{9jrddlBh5YGcR^7q2n=|21|_!IiyR
zm9EnjE;?tZKq?~?ATsm$F^2%tFvM!9dvBlHqYL5>eRogYk&KEV_waw`Fn2W&0BAc@r>x9p{8Xtr}P0kU|i{RWsmOe*}2>G*N`*<MxMS?2Ex7K
MY<KW0U*|RJ&#@(x5lU9eAUP8#Y(l>P#b8MzU#D3(hd<&`VlCzMczOqol!F65Jb-8gG_k?h!8C+gvn$?MB%|Mw67smc_$0>G-
U4*FhKVGsp=xqbO28NYikBs4<LEJQ7T}pd&`v^*C?bbNL+a!`I{iBnL)gP^<FhqJt3r$D6J-
$SN(%ODhi_zo2EcDs}tT)E1(Ph%=7(8Rtxh_uYI^1h@O?z2X<SP@rMVjKHIOCegT4w{S3=5rjPj;EBzSvqzS7R49RH<qZ5O_*(r@7m{gb4h+je
&bM=^cUxw;S;lkdctZ9Scj7>}U^%wk5Q$>%KT)DTgH5^-PgJ&;&`L15wZu@nx_*)Ri8v{<YIaRq%C0V-
7D)b}cK8(c)**k+<#*jS@`#r79Fw!Me<3voJNMDDJdQNjGJ&+#3e;!v(N;$e;%drXr+ef-
0>XQy|I&Tpzyv3Uzj3f4K9}&Wo(~CA?iLXZWih74!jRt1dleK0wv4kwNzWdCLG2O$>{+YhaBC^|FjeUI2=Tzh%$~~$Ztv`$PK989`LHb#rb-
I}pbp_4y#Im}cf$3A@Y2(20@l&E}9IGnGzyz_dB#<7FsQvN_K)FlV+Vkz;wOj?C3zgxzPmyKch)86t!v;$t8NwNKiiS2T0cW@m#8omnQKGASvI
_RxZY4>u%OmRUm`^|ZQMs&r(tthUSq~G^3}p>A-Id~Fy1K3yW2hm&%C>yhuOMUy?vJ20ebyO()o#{qJxF4=Na%C%1&5%JzF|Y!%-
{V<dFx{;bafV{a`$%y#HVTa51%AuM^ZHVrkxd6?I1vKS~jf_1*f$!4Hp|++ygm07I&(?jtFNY^4OnWiL2(@wO$D-
Tj4DvdKaV5a@LJ7zxiFt#46H2d(2NfK!2O3ysObJuC8x+XcpllXm2<sMfCxm5q^d7)R-eLbMngg!-
B<>B_1cb5050sjyRFO{+<g*m>uyTq5uO6^Z37arLNA#*7k;f-
<baEV0JEG?Xt$1x_M8JR@x%XYbV=~BtP@rQqyTKxpl|%hh>R72OUZ~!TQcm1vY-yXvt5%P4GalxYR_M^HPQL*er^xf0w;*UV!`3!u&Sd1gg0|<
?q#lf{{EDey^LoyAt^uZ&fuu?TE_4qp>#W+hzvJH^mLJ)`IdxcE!{TB<MGXnIZ|nO0$M8Z0$scRWo|ab`!>Lo_dn7u$L9Ci09u#m<w+-
D4umFw8MJxBD~d$%~hjhFTkp-
T&z;<RoY?)Rz;IjinjN3^u}C$z>%Y5^@?nJB(ggK^rc`gWe32vx@o_|yJm7>#2c2%kfZnT>Kh}HIjG<tE^<~EEvYqhp<%nyxKC03n4ysS6lsh=
@h>-L%tVZ!A&#QgMwcw0WirqGfILdP_$_}CNIWM3)-
1kGsxWX$0!~f3+gk$B<J4iD)spBGtshwdj&vL3GY<><qTkGzRFV4f{I#69!L^*^)8I?nN2gpS;(L49J#%iDzs+9FT2Ti0ViDiyQB@>El6Z{H?r
norg24;lTbH$w-
8+v%qkQK+)oGE8hPDcMDEkXXV!BXTVBY+@K!F@d<|^An>&v^0zpce4*8EAI*UuaNm8NG|Mmv=#O<4Ti3cFL!<&x#hG;qFmttuV8Xr+M79;F7lN
~}#XhaubfA`}UI3+vfVr6aPm!I_=+r;{Ts)T;~L&!R;^ERxN<E?>VJ7n%kLQcT@pw`mRfC3J0UvBzKbW?qh0C+p!{)OiD<o}62FDBO@;ikb6+N
FM1PSN#D2yfK@d-uKs7S(a>%^Q681Djj=qzXbl*lT+NZc`5uq5lw(}Kh$W=2~?;zSfAo*=LrFHkNgGnX(O-
IkrmXSBhihj+3G5S&A>EgA~_@1ct~MBl02tP$mfL{I{nc$9Xx-8xxfK;bj}iD@Xr0Pb_xefVALz)I*lj#RWKGJ=|qJ-HF5u`_pTg)IMR(Mz&NB
8BQ>7%`B$Ma(@&Q8AZe4`-
$2`lXjC6hGGAg0r+OH<Q|>m|86>WH61a?c3`crh=AJMMBvBsgtiD~UgWy~PuJ~L6c*>=_*+REQSSBbSiIH4iu0VA~!A)mc_nW5)S)Cn0d}7IC_
TX&wNUaK*SEF~@OyH%1(6(cc_pE<JM~?+&mSoq^3Y9|x{d@f&s!Zay`a|!Pr${PzwzpV#EQH!6yQ{qYgD2ZJ57oZz$M@wAo>3ivMZ+5qqtjTgx
W{pf-XGz>+8O*%29)iXhH1;NInr==ObS7ijhJF5iZ{KGp9Y0jpo`rjf0jy7GpcBb@M|{&9qiO7!5c6ehy`9Pj0IFt32UI=ooY&zP)(w|Mx?12n
apL)ms*UiPB3b)uVr2z<K#fw4b?tNT)7v|xBK{Mr<Z!NrUQ5TLSE=FBY`jyaQjSj?cu--iP`opSTo=7yXm!Vg*rEfthc+5FP_2cwqVR<p+Q_NU
mQ~_p^6HqxboY9pMGb*?<~sT(RS>w@e)@|elYHO>r)BRqH*xB5DFSlQ!W$B%m(_43I^qCFzV_Qj5Mx(88L0Hs-a-%_@NJYsZPv+@-
x6x=%oTDfNORMp@NSErZ|;$ZAx#=+*5nba~N6abf=lS6KF!S8wo<5iKOULV@E#NVMY>6HC*w)6rL=4U$=}DH*3kwxd?+j(;poo)&&HL1j;!miI
lTm3q5(`MbbY7u@o9I*S=+uel<TowXdT%wrAzL;GC|V`dB=oOIhp^Kmm;GA}aYbdxJ>7k*-6dAH8%iMd`c@=X);nf}ph00?-
(;hv!M4qMCnGiN=bs&Ca;r;aDqV!H!-aFi_BLO;_iAR(2}&q%4%m9VSs0pK#iO{jMyNOqEith#0HTYNek-&y@0O_9ZjItW9kB-
Y{5Nk&+)gRrsL`iBHP0>nze3Ms}ms86_)*#I`h5K;h(qMHB-Z7kY7xQCcZkZL#Xb#b-
)5Xa)dFjKR%73rA>=QX#hLZ|jHKL4h9dFHmDuTKWaQjW@%IZEY>LzA3P%yu!y{v=r7FglULL*`ceM_Nwpo1}a(itgJp8Z7sMY6uY^UobS3|nWV
W4XE&b}E3(xhn#+~gt`%AyTF`-^P>U+@P3kS8*P;CiTQLy0?Rwo_AHM9o?XiaXjZDyclrXLt$XW0ue-bh1ed;@-
Rcw<zoUYJd%s^1)rlmv|hsBQ|e;xU;D8|D{d9m|L8o9L0K@KH}&a#{8i|eFuO5C%hQC5J*C^8y<Ilg&IKdG=XGOj-n0pe)W+jqkAMt~r8=+cO-
F?aUbWTu8le`tQoM5u~M&kF6UtIj#l80FhlJ8L&u&#OOIKN{z!#vf~NOytP}vWR#Wt4cMZz-
3v(7`*K5MWyjS=goo7xf$1g3QLrz$OAnD$sisqwSb>WT?Xh5B6&gdLG)C$ff&FhEqaXjj{;<^r+hb83AC9#Ug+U{kUKviT39=q0S8wTxsqjW?{
@^~@++6ZTmW1ng!}~(cM1+{w#RbdoYn0Z?_P7JQx(L7D2hyaPp_T&nUa)gkx>TcJ=u#HI%2e@wK^9*aWy{8byyHH*UnRDJbd~sPmN$<?dqGt#D
_zsa*s;9x;Tp>Hji;EE*7|dMV2dN>egNDvf6qPSHhEPLL$D*Tr9jyzFiMnPQ4T|CYk4t+))sWHzaf-
BS<I^D;;BaTDgc;{L_Hi%?{Tye=pC!moMC4+xvBVh=FL@znOE!i4+C({zauxua-DS!6mOf^Z|tL(`fse!@bdZxa9$|fRt1C1<=<rbd>IMq|`E;
ITL}YelXIQ<5;TT8w7vD6b3K{62ezT*MgZ(spTtu4$lrbu$VS_r5N#v(E)opAPQaaz*V1Ug>O1YNtNRZxDSYIbkJvE4^1Hwd{0kW$omd^2r;5%
l+$pDz+!7Z4SM{ik|cD>D0}CrlCa$TC<jBw7rwdLY0Zn{jAPbMe@A;mTa?HGFy>M<gEjK6gCi$2mCbYaw2+l_pN!P*vJp+qLzmf8#08XGfya8f
U(_&btqq$YL@yCo`qSki4AKm!4ygGrW<6C&soiqax#A6V(qVK97GiIWJZX(dcAsRDOXuc;Bc9rmt`^WaDsVNo`tbKcW}}brKz{TEIT82ppLZ5~
LTch(o%Y;Qw9n#*bFOT$cU5CzA!#BWefEI=3N?`3i#q`TVCr5AORM%Ted|KB>lIj?!?WZOF-gK*0aYx?@S6^Wgj&*3DJXE4O4oxy=Abx-
f~EvMA)`JKkJ6~+-
%FNH+CLoLe*RvtWC#4xjhE=UQHvL{?~SuI#u7+}92|jxmK+QImSulP4#s)(Xuc5J5>n+bW@w*FlhAoY<Fqs}wQ%e~d^U`$>*cb}h;`SHj9nI<)
zlWMh#LhnR&%F}l;v(n1P!QkE~k!5uG09CQdQSlk{dc8&83#eO$W`KpXNYzNiZW|u4e@PW+@_Ek9ZsN25I2Ptmy23&r?0$(bjwuF5otITFud)X
*}n?>0=&>Ln$pd_koMQ5Qgk1Q~mI!Q<JEx*2en&wVki&$0ecfi&An9Ie>Y=>F{i|9vv$R*fAz;&TGR~dmu0gEt1=sQ+MZOcBCt5#CtLDh<F_>|
3&d4)%pB*i8ywgV*Utb#e!KX+BMVLwc)#a!1bRC{ytpmqLQ=v{DHvu<v^N*_lpm*&*`9Ybo?Ob9G`#>u%arRsUy5}U%6$9`@Xm?WQn_TP0~IyZ
RXhhv#iP5Ux{0<IPI>XHs(CTA(AvzJ^Zs-CYOPvCynd-
yw$$`Wy@Sc$!hBZgOmbvM2sGkj$rrV<#5qW;0nKueLs*fdZRM+kC(6A#UmWy;h_TPY_@{f<TQ-Pd*p_c6n8)n))=PFGK$bHd@gK$QU^DDXu-
tjaiY@-gmgNs?F}UfELIw(_6+5k*1uUN;>XuJwE4v<3QOT`Y`2>yEFNMWfARoWL7-?SvUl-qh3hH<lf$tkc*o8Z(&UCTi7uJe$Z*>XqdVVi-
`gZBP3dtA06N&qPm-*HXy*HVw$VgiO4Ea$@LyaJi4Rm0Gew{nT7d|oD_EILDR6-Ceof$qn+8qxQ$_<1>b-
=AsEiPH5`KrO78Hu{SMz!qQW;{)QAW7@o&-%A24B=$BaULQ=!76engr=DXIJp%G=))$O|9dYeyRr;f~UPtT-
)*!O47zy6E2iwS`#v}YJw~TU$KAr_mnZ$U0XEr+1V|f$-GfuDOxB7`PAF??&-+xI~!_0|D;e^P#f>!iM~UcMX$@n=el-~)|bQ=dVb(yk)9X^(-
B-
ftnpQnu{)0hHo?G7gXxxPTIVDtdC`0FH`E1I@d%g_V!M;zuB{^}^}R~%v_q2?K<G)=7?FGh<&+>wPujwp5++=5Uw|tLgCZ~Ir8qE_P*^*PH?Nr
IVZ^sc&Q=l|$~eAXSTSSNy6i}@^d}*l=V*u4;Nw;TcUzh}#W^nK&1Q{Gej#RN-
u1H<8aJ!_2^<oHs_T#Dvl1ngTKgdkZt||4ni_@{QFv_yIuV$G8Bj~&&xL`*+r<cnIzuBJhYL~&t%(LCau(msdFG%-
ti9fE(9EM=Dnd6qe2X{WcQBJ;*pZ$Rlih;uF>-
+$`)Y?}RLq8^Ep3IiNh|IBtfUnU8vi+0G8S1u8^6)EMr9+`Km`Vi<@O^wbLa>n>8^CEP9JBhuq1fD_W<`xDTxzItZb|fe9|;m35VrO^)WF1>;7
_aeR(ceVKIxErmfKo<PtzAb~m5wgGz6Ti9n33Y4|in=t;v|T(ta6Nx=#-
j2E=W>mE48&EVUrF$hhFC#k%N>>dE5=3dq^b#b^v{wAEfz>Bj@&*G!~c?WZ4rVMwn2?~9#=1z^TqPxRYAVg?eL*PK8^uEc5*o7^8CZ$o1yB!*F
cy}v0QuH}UYYh;5zj~QJoh9jH#Rw!SD*g54HjX{JF3jaN%?;eUX;)iGGEKA)JS-r{cN@=ANweeUpfp~z&=7J>uiy-
6({@EnaWU*AgC_zt=;WwxZ9i{nCq`tZFANIE>agN4gB*x_x1wApj|#9cs2pO}Lf-Y5ke^)iGwxOR5=DJzKiR0Y)>gP4j-
z=kC4l97Q=(88Q+lPfR!;Jm@I<=E=-6jhVla^`0B9MAoEKd3z%jf#<%J31#QUo8b1&2;!jsm4cJBnQIPgHN6z*!&H_}ZIS^~x2Ng-
F}wd#<KKo@z126BfCb9yNOZ+lo$jHK=dho>L2Qm{BQ4?C~^w^`^@{305%YeOK^dm$z;H!|L;Tqdb;?*YulaPG*f9weC^#@m28a3fjP{vdf5d>V
7-^XQ9;-+1VQMmpAAmy0c-7w;Dkv}81w=a`X-bn!f;*V+khrcR^+zH@GZ1`zIFsR&1;-
EU7CfJ|OLrc0)w=|iV?1ig}BR+8XbpkjY?Z0AES>_+J8ES4FdEkg~B_A>skU;e_V9D^L`&ut!m(2!;}sL4HFV-ZJq2p}IfM~5<^xC_3NNlEd)5
?&h~g=6st^06Q*O!DGzF9`OXT97(uH0%}39SI3_`Meem_%cloLf^M}h98GQL1OVp(L;wvPCX3)Sk_>^%TjzF;G~6S^R<z`9f?VV;9y#Z789ps6
rBO*MlP)0WF*9KB=3=CCaT`~kPOIHw?sO~#wqAi2R<BXA%|(Z=9wN0lO1Ec-~d119x6O1Xa=-
6Y&_I%trfQawAk@W1=E%xT4Lb3H_dWVV^!5JMfE9bQG!uH9;gYIRI$i~XW=ozw~6=x4B6O!H=5Xrm^zE^FkV=u-(%w|!-
uZ`&}hD?`lfuDkCmR)19};<kSb#@u8Z<Ob(wlko|glEvsevBLJQ|k4tD9F$Ejx`NV|OWqA1U1m3Q^X;7;8e=<L~4(mP^L)$N)%>A^8@tJw}>dW
x_I64_wudOw<u-7iD^iOHx)4}exS4y)k31{_l_9c`s7nvEt7ZzP-
!3d;3;gEsy(F42Oco^^`B!mteid#n>!zzC39poQ5G39}son0o62CvU&9cb^C9hjv`Nb3A&lnzGB6nzK`VZOY#kKd1uAEDudUM}k0xkVp@**{yu
<%O3dOeqx92o5;I|eJz=jto=&heI2*;2n8v2e^<^%<nMp`iQeajE;@Muu3d$2PSN1vkIIXWYwq6Zf%4Y!B=J0aW#okJ&>agf8e_L&*LNlQ6``s
Y4sCt~8mXz<A#krhUg*@Aeho(ipo18b+(_X#e@8p-
TP$iXfT36=HjK1)iGU;G%W9dqn}1P@Bt8*iVbuWW$Ut(CosHw8dvSp82Q}`f!l6!VxqUCDMV3JRu@2GgF^VWyHQQNRy@Z(L9&^Zk#NLnfeI4PO
$wE|k-
7ESNFf}I)m2KoEv);^sUc^Mxc{Y*3Jefhh=~nD}=`JkKsCOM}1g+DuE1xWA!B16I0l5l>$dX?uc*BedEYKLkXP{k8uaNY}dVG)w4Ai=8)<{{p0
mt`wtV*1{xjgS)$CzKOPdpeM%HV1B?2L}!;Zc6}jS6$(o{6V1NA%<%N2n<bhRlLozDCewJAN{&nke%Pi7Q(F>gEW5sy(G^Jm#iDy(&AS%;doEv
t%IT0Z*PE&Je;-EqxX-
h!x2I98YF*(YtN4{$(A@*F6Gq>Wu!$1MEuB6EMgG?P}pS5d^u^P|Zrp0!03LJ<_zpZTp0hGyJvxVw}dJL6{GW0lWp>m8evoVnyZ&XsFn`56>F-
P_KC;o+Cy8O}(@wALNpQ%6jKG>k7%J2g6oRhdrWom&Eg$Tomic0FuW|eTr|;qtLH!h4*2oYWgGCL&x60vYc*j-
#^Eg*L!O`W0XKUbG7#B>J>sk7}V`j^r*)Td$uGP-fD>u<wp8Jd%;fuWczbD+*kCfdSkUMgj2#vP8?V!%F24i#~%r4&`W?VF}VIK)fWd_53(iz3
R$)1VY4i~vSC_%9bu`<ZSS1q52v^jy9?F=k7>KSMw*<n4T4`X_PmwHq((eXhDzNfG(3!(ow_z8YJUEYc`fUxHs1Q<Q`FuKHpD@NoGzj>_7)RIg
%YvOI8=h*CJzro;P<OGDJ-
{!K2ApMF6N;s_ci={*41EtGW>|{ivi?FA^vRB(e4c|B2F1MbJS=agZ)CA+FuQfSgs)jD)$+rjHw|tJ}W52Mg{cTT5uXe((G<W6|4kd0GG8y+Ti
*KA%Ry70Qm479HOpGMVzE4m1j0gVH9OfBtL>%f?B3SM<9Y)ImK>IgZoRf#h(2%Sz-DhY6i^RM2ifPH5G6A<zzh|rT~W+T6MTG8REiD)_(gwin?
v{$*3ZVBe8&$Cwoq>b??zB<&lBmJf{z~6!N9Cb4V4|z~{EdV~!(TaKe-*)&-
`tu^s^&c_>%E^)84WY5t661^dYSFl~vBGanl<avKje%<B}T%*#d_D;iI+0nsWAOYBfJ8prq?Sd++mk|6&5Ae5h=(Hy%f_Ivi}3imW|C`wFhxpi
duIy&%G+P8wnbi$;05X51g;@sPq&@t_ame!X^ShDNvs?LPQ>ju>ed<&F4Qd_m>48$mT;RE((<e2#`26f?pUo}(TL7?*9K%KxRyHq-
&#RR$^rh!aS)hg7jtkQ5Rv=aJE9TlJ|EG_PRSS%UAJ>oJ~e=ZE>P+DGYkJZFS@u_N>Nhn^Chw@kb=`@1ltS>YnNXf1P*QrOf+laH&Qh3d)d(Kg
)_VRSdt9v{2<VDjZDiC#11w82xO&yW{Q1RlVVavNjMMm;}aO`fm4>e!O0&(qlkgPw+a~`}2UU%}c@?Yjzx6dX7wIN`Z=@HBy(sAdvQ1JBw#>ay
<jjg~K#Cp={D(W3)zh1+sX;8ZiCZzI9S>>Fh>)H%ivxk(a_2|(fv_B_r<RkMDTW>fJJihNeo4%c~7P(~`+n>41XYxY$)UbC@rXQw)NFdxKZn85
%Bd>VpxwsIVZL#)ngLGbxZ*_E~#9|+&PHgPyThE2tpl5C77{&xh3T6QJp}6X#c6lZAO{KzB!N0)ub@#TmUFV}HzE5}|7Yp(w5JrDl^C;jHvz-
9J4jhh$7vr9ZyGgq_<d=uI>?*HTSixt+wp!N?+~vC?LwS3I(cZZ984Y{{eQFM$PlZ@T;x?wdvj9e0uAGWL!q@M(Fw=?OE66vKkmt+=Op)Vpw|;
6!w6ayC#@ax6!vJ|>cL+b5Iui4v0C%k@QZ}(3ol(QY8s*pGblj=d8mO^d(`%BGHAi1*R$#0<dkG>ah}_{yL?dda>>=YXmCsIbF^LuTCbRn>eM(
(ac*VA(c3<O<M=y`4JMMy5#WFw0%%C*Nb<E<Xyo#g1yc2R`!`F5>rN3w)@dNgS@lQKX!IjvdQ8uk#?~G}P<WRQL9pZxa>MqgJ-
=CGQ2<<QXO{3cKtjLzga@_L>g{!r*X5*H)#x|?b?h8U{i0Ad%=Qz2!aYhG9)Xz|3dX=<_t3;-lMS#29VKK-
>*TTn;zE!GIuj$JukrrEb+jGibER50)=%_+f8?UYu@PeXZ3I~Z5Tk9Q(y+A`V{-
8e|`nOUT!+c)126Jh&UK|m{_)z4g60^NLOT`>{QH^5>JRTKie7xtdO>d$(>W8A*Y<M(*6<Nmj-kKbtyiY%OjbUQX+q6RZPJ%<_^g7l<iDsCFgn
K=lXI>nXU%!d^3FV1AqQ=lCWzj*Ake#CvkMyhz8u!ocn3}Aa2Z-
K|H?Ey*lFppx7|qvU0DG5$69PMbPJV08^yLQWM2S2ai_s%FvOld<)9GMMT5?sgLBZZ!E}a9pjXN~-
7AHNxy7H{NS<37{1h<aa7)clTuMMYudxxwW*WQuCcdQObYAj1r=+0`xzT)d=cou3B<5FtlG)Uwb`BBdTD{Nu`@W|>9>NZFm2b}16D?TUX)tR}m
XFc^~Hd^_<$nmP#T-Y^6&uAO!TaVTEc(~X3o+#c&$Eq7fG*K59!9JBO$L95`#p%MjV>Xl8$RI-SI67DGuj(X&xk(Qsbt)&J(6ZfzzgHp1J@h>o
wzY%fp~>OQNbMB9&%z5a3)?i~cEi#E;f4x-P5)GKDW-
{BuGzaIW98%K*Z>L>BEptQ3;*e9L}630OyW1;l`#L7%x05?+Iq|0v=4@ZH@(6`FPrg7yjeI;Pt{iugt&C#cRzCZ%Vv~Tt>6do4s_w5TQ+qYf?w
GUC0D)O?46bp#H|zk&Q3#;+k{jl7xAf)lO<@AWP^6`(m*PDK%d(9U^<Q94a{IyUt^;1aMi~_tqOItI$UCP8WQd26VccewN982&mttlSn=&Vk9N
#NuFJPIB9Er7rRl-Y-~gL7V+z@PIbTka<6qa`WhzuV04QchR(4X1z1P!x<ja@!1@5UAET7_!0vV^Y*<hu*6gE3O2p5Q@uF#rbx916-
a^Yd)<`E~7*%!IZ9@?pA7)X$_=4HSY2lLtW54vNO!H7ybEEa3ot&hN<)~C;3;}g~SCst*@FJboOy4#$Or`6?52a_o`n=%~VmYGtDY$!7w6b3O1
7uyA+e(Sn~y{IfkX<N2PeRY8)Z)OACl2@7%tK&5?nN=%{&JWh5`c{k&Z?QS0Z4&s_edl#a3AuMKXgcCb*+#Nl+BZdM$Hy#1v25&BC6B8n^JSSn
#WFvkz2BWZ?A0RnR9Y}6y5SD6vv`_*p6=z4?mu<m$$N#>1D=A-XnXmQ>0$sv)>9M$e_vJw<Pfw&RJIiOO?Owo1;*&{7Wowql1^_F&x3WE<tR-
<uMX2(oFG;i5i5i=7(ISC-
)x_swhB#zBFk7wpHw(BjiW(4EgWM|UhAyqTOe>dR%DsYgQfJ?jm?;4SXjFdcy)rx*kLck9Uy+yL(#Hk6cUx|G2n)qv<~zWHD&-
fOtlCV=F)E^I@a6lwyHT$ZHTa+^|Cjr8r&J7wNmI%e?GXWn;kT8uRR;V0rKHiqY#>S_i-X-JxHJp;leBo%uF`xi3|o!OWHa#xUzm7%73JAG}Ye
am)EQ8<`|JWAsW^bZiurJyrAFSZaVcAxu24-f-iQ};fUoq1F26P$9pFBt<#UM$#F-
K4!4hE97%3i*xfGmS+3(V&majy1BrL_6?{>vHyt975B{)fvozWYd2;}qHNS2)r4X|EJ-fCd5Zzq}A`BKMQ+~KB43H&Y?SvzGbaW9#&jZYTdH8`
6>9{4|X3A-pm$?Q`w`sU)>8`O`T*1$m&v5U-bW&6j>;#c~X@Y3QyH6#vN^*l;c{;pAiIVKm+u32?N!t-Y!WEn_ZVd8CJYEt^WPJHr4E4-
n!IN1-Ab`yg7sWm0{E`^65*A+-*i@_f1QrWIDcTMucOF^tdy<Ei%6j;k#a3ET+Kl|<F|knvc=C_X;VLWzQ4S-P-yX8MEqFAxS9~H-
C(UjVXpyGjwa3!zecb&~Pm{;v>Yp>y&$*aZehP3_ua+mpAjqq=&T1{9ewfhq5=<A1Q<Cj!E7v{-
xaim>@rgQ^y*Ul<3Utrlt<#Nd9UZ@5F$R)4!|_7Ptk*BDt-CvuIOJD}&;nn`mi#UzGII5uX-!{ru~V3S4)=X|hh-
>8vXM(cx@?P}Ef*K+DaWn<gc0>f4@=3YpuSdise{C!nXFi)WMiU%XQ^DFBv;vGL4Lv;L_s2EU{K*QtUk9*p`&xtR?*}DS8v7N@eAAU+m$6%$+q
{SM9{;YD?fgr7xVuStqH}cob=ea936Y0XvAiPge2HFc@x*PMLtsp0q^+I7ll&ES&RBgyx|(49$ZmsgP=?VXQx(q<C=c^+2P+E5cJKZrA=%Cgf9
~J&<JWUN};69BNIPlDeQ0btpq2Ekh^h(M0^(Q@OzXA2TixzW<g{FT|?8>&ojt!N^^M*%bzJ8ATBM-&DMG_Jbe~>c-
s|97$QpS?WeCblh#G%#$ECL$mhjTlb%Fs!#KUU`MSD>d<wyj3jXW{(I`PNP4a_2nD#Q?T%y+Q{Jm9<j3?$2UdT8NiG+M`WAG099u-
cWoqX>~t(Bg~*Wg*s-
u83tU*Mh{jwOokzjf&wEurdNkLF(ZA?|+jdziQjW8uP!zMV9aip*4Gp1J#pYnypYk0E&gvaG#Ah9}2d8_rngkkZ&3UC`zbi9B@X?DX|Co;Leq%
9FJL&{J!)wEt6TITYky;t=^1gD8Cpyhy>7;L9Y&Bl^tf(lc=tc<?x{ktA^c@Xj2^o&3n@ch{7S`SW14X&3In9TVGac?WUD$z>3IHz#`tC)d-
IshjlSb${}MVuRyL14xp&lj@r=DO;LoFIKTy>w}MvGVpu1)1t9Z%3M~)DhCw}_k>*ugnJ3|IqtM`jg09#GJ6+dtUY$?hcy0Y1_b9r$zjay1WV}
^+QaJ{h5GK~o&ML?1qXoTZ5{6}wBcD;nyd>r0*!b6B-
<><K3!mcU1#`E?!k+t_Zj9}B`!PE`ed;RM)tc`CDULoG3>X7D!~{CAIFI7{Ub0qKf@Y>ehAdGd_O3bOhnLpjzQN{D%Ix8mFF3;-
k7yylYQ6m8P?0J<8h?@?_QenAC*&4)}F5U-(jscH{%=fRoZ3BCNoA^MMf5%qqT5n3AEoSOIgi$7JJE-
hsohPE$Aj16-$)D>M#3`7?O->+(f0Fjg-S--NK?w|KM++B{BVBIq8UY-
7c#?hj{y9U<?z@`X!*ZU33AOTV;ohx#sh|_F2ZjaLh(91{{B;ts}SmG+BzVpm20U<H|}lF5ug`mr7ZHS5xb3@=jb=&Y1-
)i6!sT$k&CCRBt6OuK=&to4vzDAz$ui*B5e2opQ}D@!s1*1FV~z3g-
qaD;XX5ao@>pIi2^gVhL2>Iw$T(fI1r|%ZjKkTe$?M{U~R9j{!dvW6wi|&qqJpt2}yIEtvhSZZ`}!JfA<(#Lkb2@zK4tMj07M-
lw@4+RH*wP2w)}oi9h}6bNDDCfmWHG1Iom!?9@cHGNMbT~rw*WNCbgCNDrKOD`;S<1ptj9mgQ8h>Lp6yKGOp+os-fH?kwnVjb*vGL__asO|u!#
OA8643)hELB4Oh*2%%Kw7dAjfSi1ZATNLmgMTO&wN9N7U_q>>jx?|ml_kwH_>z&qRIRy9aRjMC3mbUETQ1Z>-=PZ6y-
vbt^6rTkJA8rV9ug4U%NB?>?CGLSh`GFjo5m0;@NOA=G}80$o+5AvEXe<PqtO4WKya}CxL@e+{NLLu{>kq_|Hs`Ge*^wj#Qy`RL-
^ka<NxOUEtmX<H-z(_RFi+_|F#(Ylg}aj|2U2QM*S^W`3EIK@n4lIe}n!ui2MU;ru<*{ME=J8Et>cTw@&>(YbXAu{5`w-
4`qt!zfAJ}jrn`d;2(@F?|+y&_&ff0SKL4GH?jZHBc~(>4fF5M5dZbS{_AfmS^fL%{{X%TJw*
"""
EXPECTED = {
  "V04_INTEGRATION.md": "47669ca8ce02daa2f4cdfc2b905e7fdb321e14704348463e389c42899aed11a3",
  "compare_runs_v04.py": "a1d81d965708c6af900d58da66ac85b4c3627c056b36d6e07f875bd585ccb155",
  "docs/ARCHITECTURE_DECISIONS_v04.md": "d92da25b5cf0c8419f7c9bd1e58fb613348eb56397631ae62f5822211ee1ab68",
  "docs/README_v04.md": "1e976914af1d354515905eab8aa40a844a2ed09cdb6401d1c588ed59a027476b",
  "docs/RESEARCH_CONTRACT_v04.md": "118562a87d1743ef4065792a68327e89e8e4938b89db8ebdedbda015421696a1",
  "docs/ablations_v04_toy.json": "e5d0cfb80468a4b0f6195af49984f02a783c6b3c83ee96fceb0e2533c256773c",
  "docs/verification_v04_source.json": "23b8de7c4674c5a56f9dacef28521fdcb546f6bb8df45a166678ee807f8eb769",
  "evaluate_adaptive_v04.py": "a0ce518b955ebf76e776f09fd6809f2ef12efdff69570f97c2a41115e13361be",
  "expansformer_v04_model.py": "9a6aa7b360ab5068494b8953bece6f0713dd8b4cf66c33c29f4b6e3509518b7c",
  "tests/test_v04.py": "e42d521c797a51339cf3b0c2ab6d662ee4cb761f1aa9a8542f66d95d7473be75",
  "train_expansformer_v04.py": "6a09df279c67bbf27a0ef9517ac0c2c964db343d532c6b8bfdb5bf0217564d2c"
}
REQUIRED = ('expansformer_v03_model.py', 'train_expansformer_v03.py',
            'train_expansformer_v01.py', 'expansformer_model/model.py',
            'corpus_builder.py', 'dataset_pipeline.py')


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def files_in_bundle() -> dict[str, bytes]:
    data = zlib.decompress(base64.b85decode(''.join(BUNDLE.split())))
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or set(names) != set(EXPECTED):
            raise ValueError('Embedded payload inventory mismatch')
        out = {}
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or name != str(path):
                raise ValueError(f'Unsafe payload path: {name}')
            body = z.read(name)
            if sha(body) != EXPECTED[name]:
                raise ValueError(f'Payload checksum mismatch: {name}')
            if name.endswith('.py'):
                compile(body, name, 'exec')
            out[name] = body
        return out


def check_project(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f'Expected existing, non-symlink directory: {root}')
    absent = [x for x in REQUIRED if not (root / x).is_file()]
    if absent:
        raise ValueError(f'Not the expected v0.3 project; missing: {absent}')


def guarded_path(root: Path, rel: str) -> Path:
    path = root / rel
    for segment in [root, *list(path.parents)[:-1]]:
        # Symlinks in parent path could direct writes outside the project.
        if segment.is_symlink():
            raise ValueError(f'Symlink in destination path: {segment}')
    if path.is_symlink():
        raise ValueError(f'Existing destination symlink: {path}')
    if not path.resolve(strict=False).is_relative_to(root.resolve()):
        raise ValueError(f'Escaping destination: {path}')
    return path


def plan(root: Path, files: dict[str, bytes]) -> tuple[list[str], list[str], list[str]]:
    new, same, conflicts = [], [], []
    for rel, body in sorted(files.items()):
        dest = guarded_path(root, rel)
        if not dest.exists():
            new.append(rel)
        elif not dest.is_file():
            conflicts.append(rel + ' (not a regular file)')
        elif sha(dest.read_bytes()) == sha(body):
            same.append(rel)
        else:
            conflicts.append(rel + ' (different contents)')
    return new, same, conflicts


def install(root: Path, files: dict[str, bytes], new: list[str]) -> None:
    # Stage all files first; do not replace existing files, even if created concurrently.
    created = []
    with tempfile.TemporaryDirectory(prefix='.v04_stage_', dir=root) as stage_dir:
        stage = Path(stage_dir)
        for rel in new:
            s = stage / rel
            s.parent.mkdir(parents=True, exist_ok=True)
            s.write_bytes(files[rel])
            if sha(s.read_bytes()) != EXPECTED[rel]:
                raise IOError(f'Staging integrity failure: {rel}')
        try:
            for rel in new:
                dest = guarded_path(root, rel)
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive create: don't overwrite a file created since preflight.
                with dest.open('xb') as f:
                    f.write((stage / rel).read_bytes())
                created.append(dest)
                if sha(dest.read_bytes()) != EXPECTED[rel]:
                    raise IOError(f'Installation integrity failure: {rel}')
        except Exception:
            for dest in reversed(created):
                # Only roll back files that we created and that still match our payload.
                if dest.is_file() and sha(dest.read_bytes()) == EXPECTED[dest.relative_to(root).as_posix()]:
                    dest.unlink()
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description='Integrate v0.4-only neural files into your current v0.3 directory; no overwrites.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Preview additions (default)')
    mode.add_argument('--apply', action='store_true', help='Install absent v0.4 files')
    mode.add_argument('--verify', action='store_true', help='Verify all v0.4 files are installed with expected hashes')
    parser.add_argument('--target', type=Path, default=Path(__file__).absolute().parent,
                        help='v0.3 project directory (default: directory containing this script)')
    args = parser.parse_args()
    root = args.target.absolute()
    try:
        check_project(root)
        files = files_in_bundle()
        new, same, conflicts = plan(root, files)
        print(f'Expansformer v0.4 additive upgrade | target: {root}')
        print(f'Add: {len(new)} | Already identical: {len(same)} | Conflicts: {len(conflicts)}')
        for item in new:
            print(f'  + {item}')
        for item in conflicts:
            print(f'  ! {item}')
        if conflicts:
            print('REFUSED: Existing modified files are not overwritten. Inspect and reconcile manually.', file=sys.stderr)
            return 2
        if args.verify:
            if new:
                print('INCOMPLETE: missing v0.4 files; run --apply', file=sys.stderr)
                return 3
            print('VERIFIED: all bundled files match SHA-256.')
            return 0
        if args.apply:
            install(root, files, new)
            remaining, matched, conflicts = plan(root, files)
            if remaining or conflicts:
                print('Post-install verification failed', file=sys.stderr)
                return 4
            print(f'INSTALLED AND VERIFIED: {len(matched)} files. Existing v0.3 files were not replaced.')
            print('Next: python -m unittest discover -s tests -q')
            print('Then: python train_expansformer_v04.py smoke')
        else:
            print('DRY RUN ONLY: No files changed. Use --apply to install.')
        return 0
    except (ValueError, OSError, zipfile.BadZipFile, zlib.error) as e:
        print(f'UPGRADE ERROR: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
