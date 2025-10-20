hanging = ['window', 'curtain', 'curtains', 'shower curtain', 'curtain rod', 'shower curtain rod']

always_supported = ['wall', 'wall hanging', 'bath walls', 'closet wall', 'closet walls', 'closet wall',
                    'closet walls', 'door wall', 'pantry wall', 'pantry walls', 'shower wall', 'shower walls',
                    'door','sliding door', 'sliding wood door', 'bathroom stall door', 'doors', 'door frame']

component = {
    'closet' : ["closet ceiling" ,"closet door","closet doorframe","closet doors" , "closet rod" ,"closet shelf" ],
    'cabinet': ['cabinet door', 'cabinet doors'],
}

added_hanging = {
    'curtain rod': ['curtain', ],
    'shower curtain rod': ['shower curtain'],
}

# word diversity
support_express = ['support']
# opp_support_express = ['resting on', 'placed on', 'on', 'supported by', 'on the top of']
opp_support_express = ['on']

embed_express = ['']
# opp_embed_express = ['embedded into', 'placed within the area of']
opp_embed_express = ['embd into']

inside_express = ['']
# opp_inside_express = ['inside', 'placed within the area of']
opp_inside_express = ['inside']

# hanging_express = ['hanging on', 'hung on']
hanging_express = ['hung on']

# close_express = ['close to', 'adjacent to', 'beside', 'next to']
close_express = ['beside']

under_express = ['above']

# above_express = ['above', 'higher than']
# below_express = ['below', 'lower than']
# I don't need these synonyms for now

above_express = ['above']
below_express = ['below']

must_support_scannetpp = ['chair', 'sofa', 'table', 'bookshelf', 'standing lamp',
                          'shoe', 'backpack', 'bag', 'mat', 'barbell','dumbbell',
                          'trash bin', 'basket', 'tv stand', 'tablet', 'mop', 'vacum cleaner']
