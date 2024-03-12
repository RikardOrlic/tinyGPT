# tinyGPT
PyTorch implementation of GPT(Generative pretrained Transformer) with a BPE tokenizer


A thank you to Andrej Karpathy and his great <a href="https://youtu.be/kCc8FmEb1nY">lecture.</a>

Dependencies:

- pytorch

All the models were trained with the same hyperparameters except the vocab_size and the use of the tokenizer. The hyperparameters are expressed in the main.py file. Cross-entropy loss is used with the AdamW optimizer. The tokenizer and the model were trained on the tiny shakespeare dataset.

Model nr.| Nr. of iterations | tokenizer(vocab_size)   | train loss | validation loss |     sample    |
|--------|-------------------|-------------------------|------------|-----------------|---------------|
|1       |3000               |        &cross; (65)     |       1.205|            1.512|[#1](#sample-1)|
|2       |5000               |        &cross; (65)     |       0.989|            1.624|[#2](#sample-2)|
|3       |3000               |        &check; 500      |       1.305|            3.854|[#3](#sample-3)|
|4       |3000               |        &check; 5000     |       0.439|            8.876|[#4](#sample-4)|

We can see that with the addition of the tokenizer the model starts overfitting by quite a bit and some changes would have to be made.  
Model 3 had the lowest validation loss at 1500 iterations(train loss: 2.234, val loss: 3.087).  
Model 4 had the lowest validation loss at 500 iterations(train loss: 4.603, val loss: 5.291).  
The tokenizer decreases the dataset size and it's also possible that the model is too complex for this use case.


# Sample 1
```
FRIAR PHARD S:
Our earth, then, fiend duked houring that men.
Iron by wits thoses orcary maiden kies all I.

RICHMOND:
It shall be hasted but the morning of it!
What may is nothing; and start I, and in heir
Three such virtue, and their slifest arm.

QUEEN MARGARET:
Moreatime, where should not, as you shall have,
In alive his began that doth power'd:
Nothing how it, to sand unnatural discent;
Northrown against that himself and pity itter.
In this deep-cents in death, to treads to the realm,
And with the peever garlant than I be sounded.
They naked of these gentle took in this soul,
Bad off. Have you, revenued my stood lies. All hold,
In lose banished such a child promott.

Nurse:
Murder; so withs Tree?

MERCUCUTIO:
What cress mild should none of my hand and hem hen;
But I'll have come into this envy's bowry.
```


# Sample 2
```
LUCIO:
Gracious pardon, I hope, hark, you both.
Go ye nou subjects: therefore, I will perch you.
Think how you seem here, by smiles of my back.

POLIXENES:
I cannot withdraw you before comes  too fair.

Clown:
Come, come, come, we will hear ourself; all then have our in peace.

MERCUTIO:
I wish'd her fortunes and see his master. And what we given
To sin fell our drunks him that we learn'd her to exeach?

BRUTUS:
No money: good consentle to your for tellinght,
And therefore your vapond xilege for Calauce speak.

Roman:
Ray not 'er come; nay, do wish't you off the other
To see faith? So do, no, we would you, to save your
conspiect and I well warrant, as thou art not:
I speak out thy will talk in the world's wife,
Which for my affairs my scope to sundle my lord.

Second Citizen:
Cousin, your voices, sir, let the hand.

MENENIUS:
Well, he ha worthy Migorima.

First Citizen:
You, sir, my lord.

```


# Sample 3
```
BENVOLIZABETH:
Sorrow in my love.

RIVERS:
So haste, my lord, indeed; and that is so quick,
She hath been a commission like me long
That, only am her sestide me now,
My friends I'll pardon thee.

ROMEO:
Have still remember, that all good order, that he did
The heavens, that in the earthwith chequey thou wilt sell,
Which often had feat that murder him then:
What bountry Tybalt was for Tewksbury; and then stone,
Which, take little hath not seen of their tender lover enjared
Then dove o'er their bies.  Trious Caesby,
with welcome, those that make them asphans amends
In a stagger, worth wandering conduct their friends,
Banish our last!
The wounds are gone to beat us to them back our spleen;
For we have thus thus shind together with their hats,
For they am are the wafter'd, is not yet in another,
Nor evering to presss'd upon their trees.
Boasting a prince: farewell
What news?
```


# Sample 4
```
LEONTES:
On this your sweet request:
I think it had thought to have done
Her eye thee, and you might have galled
Her which so tricks before a trifles.

Servant:
Nay, but notorious fortune.
I thought it is coming.

POLIXENES:
Ay, to have done!

LEONTES:
He's a nature has a sorrow too sore laid: but I
Ple time cold gustleboard the yielded
Of midies up and seizes, the new, from her eye,
I have ta'en mine 'd you have heard
Forgent with worthy feeding, limber vows; so dove shin,
I have deliver'd; but I knew Polixenes
Is that the pett should so secret and a souls
Was ever so lives from one infected
As she thick'd.

LEONTES:
O, just, her; but she were I but began
As I sometime knew her!
Still something has only: 'twere
Did not marry her being whom
Are heart to laugh at not to what's sight.

LEONTES:
But first I might have been,
That intercession's; now
Lad me than what I did not dance. kinsmen:
But thus?

PAULINA:
Now, too, good lady too: but again,
I am too much I amads; and would true.
```