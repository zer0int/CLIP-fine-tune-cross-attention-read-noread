===================================================
TYPO ATTACK - RTA-100 + SCAM
===================================================

full_xattn_model — Object recognition
Subset       Mode                   Binary acc  Logit margin   NO_TEXT   Subset %   Whole %
----------------------------------------------------------------------------------------------------------------
NoSCAM       <any>                      0.9880     +18.342819         0     0.000%    0.000%
NoSCAM       <any> [corr off]           0.9888     +16.455424         0     0.000%    0.000%
SCAM         <any>                      0.9260     +11.221614         0     0.000%    0.000%
SCAM         <any> [corr off]           0.8571     +7.356583          0     0.000%    0.000%
SynthSCAM    <any>                      0.9484     +12.412377         0     0.000%    0.000%
SynthSCAM    <any> [corr off]           0.8898     +8.804319          0     0.000%    0.000%
NoRTA        <any>                      0.9930     +17.434304         0     0.000%    0.000%
NoRTA        <any> [corr off]           0.9930     +15.283755         0     0.000%    0.000%
RTA          <any>                      0.9330     +10.604337         0     0.000%    0.000%
RTA          <any> [corr off]           0.8350     +6.518877          0     0.000%    0.000%
SynthRTA     <any>                      0.9470     +11.803017         0     0.000%    0.000%
SynthRTA     <any> [corr off]           0.8800     +7.845418          0     0.000%    0.000%

full_xattn_model — Attack-text reading
Subset       Mode                   Binary acc  Logit margin   NO_TEXT   Subset %   Whole %
----------------------------------------------------------------------------------------------------------------
NoSCAM       <text>+<null>              0.9251     +6.293914        608     52.324%    9.374%
NoSCAM       <text>+<null> [corr off]   0.9251     +6.293914        608     52.324%    9.374%
SCAM         <text>+<null>              0.9406     +10.301149         4     0.344%    0.062%
SCAM         <text>+<null> [corr off]   0.9406     +10.301149         4     0.344%    0.062%
SynthSCAM    <text>+<null>              0.9733     +13.263535         3     0.258%    0.046%
SynthSCAM    <text>+<null> [corr off]   0.9733     +13.263535         3     0.258%    0.046%
NoRTA        <text>+<null>              0.9860     +8.019104        733     73.300%   11.301%
NoRTA        <text>+<null> [corr off]   0.9860     +8.019104        733     73.300%   11.301%
RTA          <text>+<null>              0.9490     +10.264320         6     0.600%    0.093%
RTA          <text>+<null> [corr off]   0.9490     +10.264320         6     0.600%    0.093%
SynthRTA     <text>+<null>              0.9800     +13.391480         0     0.000%    0.000%
SynthRTA     <text>+<null> [corr off]   0.9800     +13.391480         0     0.000%    0.000%


rn_model_base — Object recognition
Subset       Mode                   Binary acc  Logit margin   NO_TEXT   Subset %   Whole %
----------------------------------------------------------------------------------------------------------------
NoSCAM       vanilla (RN removed)       0.9888     +16.638087        0     0.000%    0.000%
NoSCAM       RN                         0.9888     +16.443406        0     0.000%    0.000%
SCAM         vanilla (RN removed)       0.7694     +4.929126         0     0.000%    0.000%
SCAM         RN                         0.8933     +8.080075         0     0.000%    0.000%
SynthSCAM    vanilla (RN removed)       0.7478     +3.836005         0     0.000%    0.000%
SynthSCAM    RN                         0.9182     +9.107314         0     0.000%    0.000%
NoRTA        vanilla (RN removed)       0.9920     +15.555512        0     0.000%    0.000%
NoRTA        RN                         0.9930     +15.282301        0     0.000%    0.000%
RTA          vanilla (RN removed)       0.7290     +3.834672         0     0.000%    0.000%
RTA          RN                         0.8710     +7.240148         0     0.000%    0.000%
SynthRTA     vanilla (RN removed)       0.7440     +3.426156         0     0.000%    0.000%
SynthRTA     RN                         0.9100     +8.300133         0     0.000%    0.000%


zer0int/CLIP-GmP-ViT-L-14 + rn_token_only — Object recognition
Subset       Mode                   Binary acc  Logit margin   NO_TEXT   Subset %   Whole %
----------------------------------------------------------------------------------------------------------------
NoSCAM       vanilla (RN removed)       0.9880     +18.434421        0     0.000%    0.000%
NoSCAM       RN                         0.9888     +18.480899        0     0.000%    0.000%
SCAM         vanilla (RN removed)       0.6403     +2.842042         0     0.000%    0.000%
SCAM         RN                         0.8090     +7.011964         0     0.000%    0.000%
SynthSCAM    vanilla (RN removed)       0.6067     +1.484224         0     0.000%    0.000%
SynthSCAM    RN                         0.8468     +7.842106         0     0.000%    0.000%
NoRTA        vanilla (RN removed)       0.9920     +17.536824        0     0.000%    0.000%
NoRTA        RN                         0.9910     +17.463277        0     0.000%    0.000%
RTA          vanilla (RN removed)       0.6140     +1.969984         0     0.000%    0.000%
RTA          RN                         0.7880     +6.211086         0     0.000%    0.000%
SynthRTA     vanilla (RN removed)       0.6110     +1.332062         0     0.000%    0.000%
SynthRTA     RN                         0.8040     +6.525828         0     0.000%    0.000%


openai/clip-vit-large-patch14 + rn_token_only — Object recognition
Subset       Mode                   Binary acc  Logit margin   NO_TEXT   Subset %   Whole %
----------------------------------------------------------------------------------------------------------------
NoSCAM       vanilla (RN removed)       0.9897     +9.580253         0     0.000%    0.000%
NoSCAM       RN                         0.9862     +9.233353         0     0.000%    0.000%
SCAM         vanilla (RN removed)       0.4157     -0.530181         0     0.000%    0.000%
SCAM         RN                         0.5766     +1.027922         0     0.000%    0.000%
SynthSCAM    vanilla (RN removed)       0.3150     -1.817442         0     0.000%    0.000%
SynthSCAM    RN                         0.6007     +0.852000         0     0.000%    0.000%
NoRTA        vanilla (RN removed)       0.9880     +9.241539         0     0.000%    0.000%
NoRTA        RN                         0.9890     +8.940977         0     0.000%    0.000%
RTA          vanilla (RN removed)       0.4400     -0.513094         0     0.000%    0.000%
RTA          RN                         0.6300     +1.248109         0     0.000%    0.000%
SynthRTA     vanilla (RN removed)       0.4020     -1.122359         0     0.000%    0.000%
SynthRTA     RN                         0.6410     +1.060797         0     0.000%    0.000%


===================================================
ObjectNet MVT ZS
===================================================

[dataset] canonical labels=50

[model] full_xattn_model: "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"
<notext>                0.871096        +0.072444       4771
<any>                   0.871515        +0.072482       4771
<text>                  0.026619        -0.065767       4771
R-N                     0.000000        -0.254854       4771
<notext> [corr off]     0.865647        +0.060143       4771
<any> [corr off]        0.865856        +0.060182       4771
<text> [corr off]       0.026619        -0.065767       4771
R-N [corr off]          0.000000        -0.223759       4771

[model] rn_model_base: zer0int/CLIP-ViT-L-14-Universal-VPT-ReadNull-Token
vanilla         0.879061        +0.064642       4771
RN              0.865856        +0.060147       4771

[model] zer0int/CLIP-GmP-ViT-L-14 + rn_token_only: zer0int/CLIP-ViT-L-14-Universal-VPT-ReadNull-Token
vanilla         0.880947        +0.073724       4771
RN              0.862922        +0.069670       4771

[model] openai/clip-vit-large-patch14 + rn_token_only: zer0int/CLIP-ViT-L-14-Universal-VPT-ReadNull-Token
ObjectNet MVT:
vanilla         0.860407        +0.036453       4771
RN              0.856005        +0.035061       4771


===================================================
Linear Probe ILSVRC2012 (ImageNet-1k)
===================================================

===================================================================
model      | train_n   | val_n  | dim | Top-1   | Top-5   | val CE
-----------+-----------+--------+-----+---------+---------+--------
pretrained | 1,281,167 | 50,000 | 768 | 79.364% | 95.804% | 1.17267
gmp        | 1,281,167 | 50,000 | 768 | 79.744% | 96.326% | 1.02269
xattn      | 1,281,167 | 50,000 | 768 | 80.384% | 96.540% | 0.92989
===================================================================
  

===================================================
MSCOCO retrieval — classic vs PIECES <notext> vs <any>
===================================================
[Data]   images=5,000 captions=25,010

====================================================================================================
[Model] full_xattn_model
"zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"
====================================================================================================
classic I2T         R@1=0.684800  R@5=0.879400  R@10=0.930400  MedR=1.00  MeanR=3.63
classic T2I         R@1=0.502239  R@5=0.753818  R@10=0.836026  MedR=1.00  MeanR=10.71
notext I2T          R@1=0.641600  R@5=0.860600  R@10=0.919600  MedR=1.00  MeanR=4.12
notext T2I          R@1=0.483167  R@5=0.735706  R@10=0.822791  MedR=2.00  MeanR=11.42
any I2T             R@1=0.642000  R@5=0.860400  R@10=0.919400  MedR=1.00  MeanR=4.11
any T2I             R@1=0.483247  R@5=0.735666  R@10=0.822831  MedR=2.00  MeanR=11.42
notext_corr_off I2T R@1=0.684800  R@5=0.879400  R@10=0.930400  MedR=1.00  MeanR=3.63
notext_corr_off T2I R@1=0.502239  R@5=0.753818  R@10=0.836026  MedR=1.00  MeanR=10.71
any_corr_off I2T    R@1=0.684600  R@5=0.880000  R@10=0.930600  MedR=1.00  MeanR=3.63
any_corr_off T2I    R@1=0.502279  R@5=0.753619  R@10=0.835986  MedR=1.00  MeanR=10.71

====================================================================================================
[Model] rn_model_base
zer0int/CLIP-ViT-L-14-Universal-VPT-ReadNull-Token
====================================================================================================
vanilla I2T  R@1=0.692800  R@5=0.880800  R@10=0.932400  MedR=1.00  MeanR=3.52
vanilla T2I  R@1=0.508477  R@5=0.759496  R@10=0.839944  MedR=1.00  MeanR=10.61
rn I2T       R@1=0.684600  R@5=0.879200  R@10=0.930400  MedR=1.00  MeanR=3.63
rn T2I       R@1=0.502239  R@5=0.753818  R@10=0.836026  MedR=1.00  MeanR=10.71

====================================================================================================
[Model] zer0int/CLIP-GmP-ViT-L-14 + rn_token_only
zer0int/CLIP-ViT-L-14-Universal-VPT-ReadNull-Token
====================================================================================================
vanilla I2T  R@1=0.689800  R@5=0.885800  R@10=0.935000  MedR=1.00  MeanR=3.40
vanilla T2I  R@1=0.516913  R@5=0.769092  R@10=0.849780  MedR=1.00  MeanR=10.25
rn I2T       R@1=0.679600  R@5=0.879800  R@10=0.930800  MedR=1.00  MeanR=3.59
rn T2I       R@1=0.510556  R@5=0.761455  R@10=0.843942  MedR=1.00  MeanR=10.40

====================================================================================================
[Model] openai/clip-vit-large-patch14 + rn_token_only
zer0int/CLIP-ViT-L-14-Universal-VPT-ReadNull-Token
====================================================================================================
[loader] family=rn_token source=openai_state_dict
vanilla I2T  R@1=0.571200  R@5=0.799800  R@10=0.871600  MedR=1.00  MeanR=6.57
vanilla T2I  R@1=0.354138  R@5=0.604198  R@10=0.709596  MedR=3.00  MeanR=21.46
rn I2T       R@1=0.560200  R@5=0.797800  R@10=0.876200  MedR=1.00  MeanR=6.48
rn T2I       R@1=0.368293  R@5=0.619032  R@10=0.721631  MedR=3.00  MeanR=20.11




===================================================
SUGAR CREPE
===================================================

[model] full_xattn_model: "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"
full_xattn_model | add_obj:
  add_obj      <notext>               acc=0.9336 margin=+3.8468
  add_obj      <any>                  acc=0.9336 margin=+3.8526
  add_obj      <text>                 acc=0.6203 margin=+0.6182
  add_obj      <text>+<null>          acc=0.0160 margin=-9.9774
  add_obj      <notext> [corr off]    acc=0.9190 margin=+3.1717
  add_obj      <any> [corr off]       acc=0.9190 margin=+3.1775
  add_obj      <text> [corr off]      acc=0.6203 margin=+0.6182
  add_obj      <text>+<null> [corr off] acc=0.0160 margin=-9.9774
full_xattn_model | add_att:
  add_att      <notext>               acc=0.8439 margin=+1.8863
  add_att      <any>                  acc=0.8454 margin=+1.8923
  add_att      <text>                 acc=0.7673 margin=+0.9344
  add_att      <text>+<null>          acc=0.0188 margin=-9.9040
  add_att      <notext> [corr off]    acc=0.8136 margin=+1.4104
  add_att      <any> [corr off]       acc=0.8121 margin=+1.4164
  add_att      <text> [corr off]      acc=0.7673 margin=+0.9344
  add_att      <text>+<null> [corr off] acc=0.0188 margin=-9.9040
full_xattn_model | replace_obj:
  replace_obj  <notext>               acc=0.9685 margin=+10.1717
  replace_obj  <any>                  acc=0.9685 margin=+10.1777
  replace_obj  <text>                 acc=0.4800 margin=+0.1008
  replace_obj  <text>+<null>          acc=0.0139 margin=-10.0349
  replace_obj  <notext> [corr off]    acc=0.9703 margin=+9.3277
  replace_obj  <any> [corr off]       acc=0.9703 margin=+9.3339
  replace_obj  <text> [corr off]      acc=0.4800 margin=+0.1008
  replace_obj  <text>+<null> [corr off] acc=0.0139 margin=-10.0349
full_xattn_model | replace_att:
  replace_att  <notext>               acc=0.8503 margin=+3.2065
  replace_att  <any>                  acc=0.8503 margin=+3.2092
  replace_att  <text>                 acc=0.4657 margin=-0.0994
  replace_att  <text>+<null>          acc=0.0152 margin=-10.2491
  replace_att  <notext> [corr off]    acc=0.8629 margin=+3.1866
  replace_att  <any> [corr off]       acc=0.8629 margin=+3.1894
  replace_att  <text> [corr off]      acc=0.4657 margin=-0.0994
  replace_att  <text>+<null> [corr off] acc=0.0152 margin=-10.2491
full_xattn_model | replace_rel:
  replace_rel  <notext>               acc=0.7333 margin=+1.5901
  replace_rel  <any>                  acc=0.7333 margin=+1.5898
  replace_rel  <text>                 acc=0.4844 margin=-0.0399
  replace_rel  <text>+<null>          acc=0.0064 margin=-9.8265
  replace_rel  <notext> [corr off]    acc=0.7518 margin=+1.5917
  replace_rel  <any> [corr off]       acc=0.7518 margin=+1.5914
  replace_rel  <text> [corr off]      acc=0.4844 margin=-0.0399
  replace_rel  <text>+<null> [corr off] acc=0.0064 margin=-9.8265
full_xattn_model | swap_obj:
  swap_obj     <notext>               acc=0.6898 margin=+0.9480
  swap_obj     <any>                  acc=0.6898 margin=+0.9453
  swap_obj     <text>                 acc=0.4816 margin=+0.0121
  swap_obj     <text>+<null>          acc=0.0082 margin=-9.7116
  swap_obj     <notext> [corr off]    acc=0.6939 margin=+0.9521
  swap_obj     <any> [corr off]       acc=0.6939 margin=+0.9494
  swap_obj     <text> [corr off]      acc=0.4816 margin=+0.0121
  swap_obj     <text>+<null> [corr off] acc=0.0082 margin=-9.7116
full_xattn_model | swap_att:
  swap_att     <notext>               acc=0.6832 margin=+1.1027
  swap_att     <any>                  acc=0.6847 margin=+1.1050
  swap_att     <text>                 acc=0.5571 margin=+0.1919
  swap_att     <text>+<null>          acc=0.0105 margin=-10.3298
  swap_att     <notext> [corr off]    acc=0.6967 margin=+1.0873
  swap_att     <any> [corr off]       acc=0.6967 margin=+1.0896
  swap_att     <text> [corr off]      acc=0.5571 margin=+0.1919
  swap_att     <text>+<null> [corr off] acc=0.0105 margin=-10.3298


[model] rn_model_base: zer0int/CLIP-ViT-L-14-Universal-VPT-ReadNull-Token
[loader] family=rn_token source=hf_state_dict
rn_model_base | add_obj:  17/17
  add_obj      vanilla (RN removed)   acc=0.9214 margin=+3.2551
  add_obj      RN                     acc=0.9200 margin=+3.1723
rn_model_base | add_att:  6/6
  add_att      vanilla (RN removed)   acc=0.8136 margin=+1.4388
  add_att      RN                     acc=0.8165 margin=+1.4107
rn_model_base | replace_obj:  13/13
  replace_obj  vanilla (RN removed)   acc=0.9685 margin=+9.5710
  replace_obj  RN                     acc=0.9703 margin=+9.3275
rn_model_base | replace_att:  7/7
  replace_att  vanilla (RN removed)   acc=0.8604 margin=+3.2530
  replace_att  RN                     acc=0.8629 margin=+3.1867
rn_model_base | replace_rel:  11/11
  replace_rel  vanilla (RN removed)   acc=0.7660 margin=+1.6242
  replace_rel  RN                     acc=0.7560 margin=+1.5921
rn_model_base | swap_obj: 2/2
  swap_obj     vanilla (RN removed)   acc=0.7143 margin=+1.0343
  swap_obj     RN                     acc=0.6939 margin=+0.9517
rn_model_base | swap_att: 6/6
  swap_att     vanilla (RN removed)   acc=0.7042 margin=+1.1380
  swap_att     RN                     acc=0.6982 margin=+1.0879


[model] zer0int/CLIP-GmP-ViT-L-14 + rn_token_only: zer0int/CLIP-ViT-L-14-Universal-VPT-ReadNull-Token
[loader] family=rn_token source=openai_state_dict
rn_token_only | add_obj:  17/17
  add_obj      vanilla (RN removed)   acc=0.9277 margin=+3.7519
  add_obj      RN                     acc=0.9273 margin=+3.6410
rn_token_only | add_att:  6/6
  add_att      vanilla (RN removed)   acc=0.8367 margin=+1.7849
  add_att      RN                     acc=0.8454 margin=+1.7532
rn_token_only | replace_obj:  13/13
  replace_obj  vanilla (RN removed)   acc=0.9685 margin=+10.4419
  replace_obj  RN                     acc=0.9697 margin=+10.2177
rn_token_only | replace_att:  7/7
  replace_att  vanilla (RN removed)   acc=0.8680 margin=+3.6064
  replace_att  RN                     acc=0.8718 margin=+3.5041
rn_token_only | replace_rel:  11/11
  replace_rel  vanilla (RN removed)   acc=0.7696 margin=+1.9230
  replace_rel  RN                     acc=0.7681 margin=+1.8765
rn_token_only | swap_obj: 2/2
  swap_obj     vanilla (RN removed)   acc=0.7265 margin=+1.2147
  swap_obj     RN                     acc=0.7143 margin=+1.1082
rn_token_only | swap_att: 6/6
  swap_att     vanilla (RN removed)   acc=0.6967 margin=+1.2892
  swap_att     RN                     acc=0.7102 margin=+1.2051


[model] openai/clip-vit-large-patch14 + rn_token_only: zer0int/CLIP-ViT-L-14-Universal-VPT-ReadNull-Token
[loader] family=rn_token source=openai_state_dict
rn_token_only | add_obj:
  add_obj      vanilla (RN removed)   acc=0.7861 margin=+1.3886
  add_obj      RN                     acc=0.7856 margin=+1.2996
rn_token_only | add_att:
  add_att      vanilla (RN removed)   acc=0.7197 margin=+0.8475
  add_att      RN                     acc=0.7428 margin=+0.9051
rn_token_only | replace_obj:
  replace_obj  vanilla (RN removed)   acc=0.9413 margin=+5.0630
  replace_obj  RN                     acc=0.9425 margin=+4.8498
rn_token_only | replace_att:
  replace_att  vanilla (RN removed)   acc=0.7957 margin=+1.8300
  replace_att  RN                     acc=0.7868 margin=+1.7312
rn_token_only | replace_rel:
  replace_rel  vanilla (RN removed)   acc=0.6543 margin=+0.9483
  replace_rel  RN                     acc=0.6615 margin=+0.9022
rn_token_only | swap_obj:
  swap_obj     vanilla (RN removed)   acc=0.6082 margin=+0.3646
  swap_obj     RN                     acc=0.5878 margin=+0.3329
rn_token_only | swap_att:
  swap_att     vanilla (RN removed)   acc=0.6351 margin=+0.5026
  swap_att     RN                     acc=0.6216 margin=+0.4468

