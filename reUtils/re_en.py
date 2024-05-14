from openie import StanfordOpenIE

class ReEn:
    def __init__(self):
        self.properties = {
        'openie.affinity_probability_cap': 7 / 8,
        }
    def deal(self, text):
        self.cur = []
        with StanfordOpenIE(properties=self.properties) as client:
            self.content = text
            for triple in client.annotate(text):
                self.cur.append(triple)
        return self.cur
    
if __name__ == '__main__':
    a = ReEn()
    print(a.deal('Barack Obama was born in Hawaii. Richard Manning wrote this sentence.'))

# # https://stanfordnlp.github.io/CoreNLP/openie.html#api
# # Default value of openie.affinity_probability_cap was 1/3.
# properties = {
#     'openie.affinity_probability_cap': 7 / 8,
# }

# with StanfordOpenIE(properties=properties) as client:
#     # text = 'Barack Obama was born in Hawaii. Richard Manning wrote this sentence.'
#     # text = 'The CNN algorithm is mainly used to realize the recognition and feature extraction of target images'
    
#     # text = "In order to quantitatively evaluate the parameters of the surrounding rock of ice-water piles, a rapid grading method for the surrounding rock of ice-water piles is proposed, and the deformation control standard of ice-water piles is given, which mainly includes: (1) grading the surrounding rock at the design stage according to the relevant calculation method of the specification, combining the relevant foreign specifications with the basic characteristics of ice-water piles and geomorphological features; (2) grading the surrounding rock at the construction stage according to the compact state of the soil, the content of fines, and the water content of fines. (2) Classify the surrounding rock at the construction stage according to the soil compactness and fine-grain content and fine-grain water content, and classify the ice-water accumulator into 3 basic classes and 4 subclasses; (3) Propose the final surrounding rock classification according to the cementation between the groundwater and the surrounding rock; (4) Based on the classification of the surrounding rock of the ice-water accumulator tunnel, calculate the characteristic curves of the surrounding rock and the permissible deformation of the tunnel under different levels of the surrounding rock, and then put forward the method of determining the support force of the tunnel based on the permissible deformation. The research results can provide reference for the classification of surrounding rocks and the determination of support force in ice-water accumulation tunnels."
#     text = "LuXun住在上海"


#     print('Text: %s.' % text)
#     for triple in client.annotate(text):
#         print('|-', triple)
