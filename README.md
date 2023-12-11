### Framing the Problem

**Our exploratory data analysis on this dataset can be found**: [here](https://the-bruz.github.io/Recipes-and-Ratings-Analysis/)  

In this project, the calories of recipes will be predicted based on the other nutrition of recipes. The model used is the linear regression model, and it will be a regression task.  

The response variable is the amount of calories of a recipe. It was chosen because in some cases, the calories are not labeled clearly for foods, and it will be helpful if people can predict it and establish a healthy diet plan.  

To examine the model, two metrics will be used: R-square and RMSE, which are the common choices for a regression task and can reflect the performance of the model appropriately. These two values are derived with the following formulas:  

```HTML
<mjx-container class="MathJax" jax="SVG" style="position: relative;"><svg xmlns="http://www.w3.org/2000/svg" width="18.105ex" height="3.757ex" role="img" focusable="false" viewBox="0 -1106.5 8002.6 1660.6" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" style="vertical-align: -1.254ex;"><defs><path id="MJX-97-TEX-I-1D445" d="M230 637Q203 637 198 638T193 649Q193 676 204 682Q206 683 378 683Q550 682 564 680Q620 672 658 652T712 606T733 563T739 529Q739 484 710 445T643 385T576 351T538 338L545 333Q612 295 612 223Q612 212 607 162T602 80V71Q602 53 603 43T614 25T640 16Q668 16 686 38T712 85Q717 99 720 102T735 105Q755 105 755 93Q755 75 731 36Q693 -21 641 -21H632Q571 -21 531 4T487 82Q487 109 502 166T517 239Q517 290 474 313Q459 320 449 321T378 323H309L277 193Q244 61 244 59Q244 55 245 54T252 50T269 48T302 46H333Q339 38 339 37T336 19Q332 6 326 0H311Q275 2 180 2Q146 2 117 2T71 2T50 1Q33 1 33 10Q33 12 36 24Q41 43 46 45Q50 46 61 46H67Q94 46 127 49Q141 52 146 61Q149 65 218 339T287 628Q287 635 230 637ZM630 554Q630 586 609 608T523 636Q521 636 500 636T462 637H440Q393 637 386 627Q385 624 352 494T319 361Q319 360 388 360Q466 361 492 367Q556 377 592 426Q608 449 619 486T630 554Z"></path><path id="MJX-97-TEX-N-32" d="M109 429Q82 429 66 447T50 491Q50 562 103 614T235 666Q326 666 387 610T449 465Q449 422 429 383T381 315T301 241Q265 210 201 149L142 93L218 92Q375 92 385 97Q392 99 409 186V189H449V186Q448 183 436 95T421 3V0H50V19V31Q50 38 56 46T86 81Q115 113 136 137Q145 147 170 174T204 211T233 244T261 278T284 308T305 340T320 369T333 401T340 431T343 464Q343 527 309 573T212 619Q179 619 154 602T119 569T109 550Q109 549 114 549Q132 549 151 535T170 489Q170 464 154 447T109 429Z"></path><path id="MJX-97-TEX-N-3D" d="M56 347Q56 360 70 367H707Q722 359 722 347Q722 336 708 328L390 327H72Q56 332 56 347ZM56 153Q56 168 72 173H708Q722 163 722 153Q722 140 707 133H70Q56 140 56 153Z"></path><path id="MJX-97-TEX-N-31" d="M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z"></path><path id="MJX-97-TEX-N-2212" d="M84 237T84 250T98 270H679Q694 262 694 250T679 230H98Q84 237 84 250Z"></path><path id="MJX-97-TEX-SO-2211" d="M61 748Q64 750 489 750H913L954 640Q965 609 976 579T993 533T999 516H979L959 517Q936 579 886 621T777 682Q724 700 655 705T436 710H319Q183 710 183 709Q186 706 348 484T511 259Q517 250 513 244L490 216Q466 188 420 134T330 27L149 -187Q149 -188 362 -188Q388 -188 436 -188T506 -189Q679 -189 778 -162T936 -43Q946 -27 959 6H999L913 -249L489 -250Q65 -250 62 -248Q56 -246 56 -239Q56 -234 118 -161Q186 -81 245 -11L428 206Q428 207 242 462L57 717L56 728Q56 744 61 748Z"></path><path id="MJX-97-TEX-N-28" d="M94 250Q94 319 104 381T127 488T164 576T202 643T244 695T277 729T302 750H315H319Q333 750 333 741Q333 738 316 720T275 667T226 581T184 443T167 250T184 58T225 -81T274 -167T316 -220T333 -241Q333 -250 318 -250H315H302L274 -226Q180 -141 137 -14T94 250Z"></path><path id="MJX-97-TEX-I-1D466" d="M21 287Q21 301 36 335T84 406T158 442Q199 442 224 419T250 355Q248 336 247 334Q247 331 231 288T198 191T182 105Q182 62 196 45T238 27Q261 27 281 38T312 61T339 94Q339 95 344 114T358 173T377 247Q415 397 419 404Q432 431 462 431Q475 431 483 424T494 412T496 403Q496 390 447 193T391 -23Q363 -106 294 -155T156 -205Q111 -205 77 -183T43 -117Q43 -95 50 -80T69 -58T89 -48T106 -45Q150 -45 150 -87Q150 -107 138 -122T115 -142T102 -147L99 -148Q101 -153 118 -160T152 -167H160Q177 -167 186 -165Q219 -156 247 -127T290 -65T313 -9T321 21L315 17Q309 13 296 6T270 -6Q250 -11 231 -11Q185 -11 150 11T104 82Q103 89 103 113Q103 170 138 262T173 379Q173 380 173 381Q173 390 173 393T169 400T158 404H154Q131 404 112 385T82 344T65 302T57 280Q55 278 41 278H27Q21 284 21 287Z"></path><path id="MJX-97-TEX-I-1D456" d="M184 600Q184 624 203 642T247 661Q265 661 277 649T290 619Q290 596 270 577T226 557Q211 557 198 567T184 600ZM21 287Q21 295 30 318T54 369T98 420T158 442Q197 442 223 419T250 357Q250 340 236 301T196 196T154 83Q149 61 149 51Q149 26 166 26Q175 26 185 29T208 43T235 78T260 137Q263 149 265 151T282 153Q302 153 302 143Q302 135 293 112T268 61T223 11T161 -11Q129 -11 102 10T74 74Q74 91 79 106T122 220Q160 321 166 341T173 380Q173 404 156 404H154Q124 404 99 371T61 287Q60 286 59 284T58 281T56 279T53 278T49 278T41 278H27Q21 284 21 287Z"></path><path id="MJX-97-TEX-N-5E" d="M112 560L249 694L257 686Q387 562 387 560L361 531Q359 532 303 581L250 627L195 580Q182 569 169 557T148 538L140 532Q138 530 125 546L112 560Z"></path><path id="MJX-97-TEX-N-29" d="M60 749L64 750Q69 750 74 750H86L114 726Q208 641 251 514T294 250Q294 182 284 119T261 12T224 -76T186 -143T145 -194T113 -227T90 -246Q87 -249 86 -250H74Q66 -250 63 -250T58 -247T55 -238Q56 -237 66 -225Q221 -64 221 250T66 725Q56 737 55 738Q55 746 60 749Z"></path><path id="MJX-97-TEX-N-2013" d="M0 248V285H499V248H0Z"></path></defs><g stroke="currentColor" fill="currentColor" stroke-width="0" transform="scale(1,-1)"><g data-mml-node="math"><g data-mml-node="msup"><g data-mml-node="mi"><use data-c="1D445" xlink:href="#MJX-97-TEX-I-1D445"></use></g><g data-mml-node="mn" transform="translate(792,363) scale(0.707)"><use data-c="32" xlink:href="#MJX-97-TEX-N-32"></use></g></g><g data-mml-node="mo" transform="translate(1473.3,0)"><use data-c="3D" xlink:href="#MJX-97-TEX-N-3D"></use></g><g data-mml-node="mn" transform="translate(2529.1,0)"><use data-c="31" xlink:href="#MJX-97-TEX-N-31"></use></g><g data-mml-node="mo" transform="translate(3251.3,0)"><use data-c="2212" xlink:href="#MJX-97-TEX-N-2212"></use></g><g data-mml-node="mfrac" transform="translate(4251.6,0)"><g data-mml-node="mrow" transform="translate(220,516.8) scale(0.707)"><g data-mml-node="mo"><use data-c="2211" xlink:href="#MJX-97-TEX-SO-2211"></use></g><g data-mml-node="mo" transform="translate(1056,0)"><use data-c="28" xlink:href="#MJX-97-TEX-N-28"></use></g><g data-mml-node="msub" transform="translate(1445,0)"><g data-mml-node="mi"><use data-c="1D466" xlink:href="#MJX-97-TEX-I-1D466"></use></g><g data-mml-node="TeXAtom" transform="translate(523,-150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mi"><use data-c="1D456" xlink:href="#MJX-97-TEX-I-1D456"></use></g></g></g><g data-mml-node="mo" transform="translate(2262,0)"><use data-c="2212" xlink:href="#MJX-97-TEX-N-2212"></use></g><g data-mml-node="msub" transform="translate(3040,0)"><g data-mml-node="TeXAtom" data-mjx-texclass="ORD"><g data-mml-node="mover"><g data-mml-node="mi"><use data-c="1D466" xlink:href="#MJX-97-TEX-I-1D466"></use></g><g data-mml-node="mo" transform="translate(300.6,16) translate(-250 0)"><use data-c="5E" xlink:href="#MJX-97-TEX-N-5E"></use></g></g></g><g data-mml-node="TeXAtom" transform="translate(523,-150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mi"><use data-c="1D456" xlink:href="#MJX-97-TEX-I-1D456"></use></g></g></g><g data-mml-node="msup" transform="translate(3856.9,0)"><g data-mml-node="mo"><use data-c="29" xlink:href="#MJX-97-TEX-N-29"></use></g><g data-mml-node="mn" transform="translate(422,363) scale(0.707)"><use data-c="32" xlink:href="#MJX-97-TEX-N-32"></use></g></g></g><g data-mml-node="mrow" transform="translate(325.7,-377.4) scale(0.707)"><g data-mml-node="mo"><use data-c="2211" xlink:href="#MJX-97-TEX-SO-2211"></use></g><g data-mml-node="mo" transform="translate(1056,0)"><use data-c="28" xlink:href="#MJX-97-TEX-N-28"></use></g><g data-mml-node="mi" transform="translate(1445,0)"><use data-c="1D466" xlink:href="#MJX-97-TEX-I-1D466"></use></g><g data-mml-node="TeXAtom" data-mjx-texclass="ORD" transform="translate(1935,0)"><g data-mml-node="mi"><use data-c="1D456" xlink:href="#MJX-97-TEX-I-1D456"></use></g></g><g data-mml-node="mo" transform="translate(2280,0)"><use data-c="2212" xlink:href="#MJX-97-TEX-N-2212"></use></g><g data-mml-node="mover" transform="translate(3058,0)"><g data-mml-node="mi" transform="translate(5,0)"><use data-c="1D466" xlink:href="#MJX-97-TEX-I-1D466"></use></g><g data-mml-node="mo" transform="translate(0,374)"><use data-c="2013" xlink:href="#MJX-97-TEX-N-2013"></use></g></g><g data-mml-node="msup" transform="translate(3558,0)"><g data-mml-node="mo"><use data-c="29" xlink:href="#MJX-97-TEX-N-29"></use></g><g data-mml-node="mn" transform="translate(422,289) scale(0.707)"><use data-c="32" xlink:href="#MJX-97-TEX-N-32"></use></g></g></g><rect width="3511" height="60" x="120" y="220"></rect></g></g></g></svg></mjx-container><script type="math/tex">R^2=1−\frac{∑(y_{i}−\hat{y}_{i})^2}{∑(y{i}−\overline{y})^2}</script>  
```

$RMSE = \sqrt{\frac{1}{n}\Sigma_{i=1}^{n}{\Big(\frac{y_i -\hat{y}_i}{\sigma_i}\Big)^2}}$  

MAE is not considered since it is not a standardized metric and thus cannot reflect the performance of the model well.  

There is no clearly sequence relations between calories and other nutrition, so we can use the features (nutrition info) to predict the response variable (calories).  

### Baseline Model

In the baseline model, two features are used:  

* protein: quantitative variable  
* sugar: quantitative variable  

Since all features are quantitative, there is no encoding performed for the model.  

The baseline model is a linear regression model, with the two features above as the input and the calories as the output.  

The final scores the baseline model received are:  

| metric   | value              |
| -------- | ------------------ |
| r-square | 0.6891289257542752 |
| rmse     | 321.19566487234107 |

The r-square of the model is below 0.7, so the baseline model is considered bad and will be modified further to gain better results.

### Final Model

There are two more features added to the model:  

* total_fat: quantitative variable  
* carbohydrates: quantitative variable  

Again, since all features are quantitative, there is no encoding performed for the model. These two features are added since they are considered a great implication to the amount of calories.  

The final model is a pipeline consisted of three parts:  

* 'std': a sklearn `StandardScaler()` to standardize all features, so that the final model will be less complicated.  
* 'poly': a sklearn `PolynomialFeatures()` which allows the model to predict calories based on the powers of the features.  
* 'lin_reg': a sklearn `LinearRegression()`, the main predictor which is suitable to perform a regression task with quantitative features.  

The hyperparameter tuned in this model is the degree of `PolynomialFeatures()`. Sklearn `GridSearchCV` is used to fine-tune the hyperparameter, which searches the best degree between 1 and 4. By the result of `GridSearchCV`, the best hyperparameter for this model is `degree=1`.  

The final scores the baseline model received are:  

| metric   | value              |
| -------- | ------------------ |
| r-square | 0.9951338041841551 |
| rmse     | 40.18595638511328  |

The performance of the final model is hugely improved from the baseline model, with a r-square of 0.99 (almost optimal) and a rmse of around 40 from previously 321.  

### Fairness Analysis

The two groups chosen are recipes with high sugar (more than median) and the ones with low sugar (less than or equal to median). To evaluate the fairness of the final model, a permutation test is run with the following configurations:  

* Null hypothesis: The final model performs the same for recipes with high sugar and low sugar.  

* Alt hypothesis: The final model performs differently for recipes with high sugar and low sugar.  
* Evaluation metric: RMSE

* Test statistics: The absolute difference in the rmse's for two groups.  

* Significant Value: 5%  

The p-value for the permutation test with 1000 repetitions is 0.002 (0.2%), which is less than the significance level. Therefore, the null hypothesis is rejected, and it is more likely that the final model performs differently on recipes with high sugar and low sugar.
