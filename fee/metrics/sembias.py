import numpy as np
from numpy import dot
from numpy.linalg import norm

SEMBIAS_DATA = """priest:nun	dog:bitch	book:magazine	doctor:nurse
priest:nun	cup:lid	algebra:geometry	manager:secretary
priest:nun	sugar:flour	pen:pencil	chef:baker
priest:nun	dogwood:elm	fiction:poetry	guard:cashier
priest:nun	fiction:poetry	shirt:blouse	programmer:homemaker
priest:nun	sugar:flour	bolt:nut	leader:assistant
priest:nun	book:magazine	pillow:mattress	researcher:librarians
priest:nun	duck:goose	flour:sugar	dentist:optometrist
priest:nun	dog:bitch	cat:dog	dispatcher:tailor
priest:nun	bolt:nut	record:cassette	engineer:dancer
priest:nun	couch:recliner	duck:goose	lawyer:clerk
priest:nun	pencil:pen	dog:cat	janitor:housekeeper
priest:nun	cup:lid	couch:recliner	developer:counselor
priest:nun	puppy:kitten	shirt:blouse	judge:stenographer
priest:nun	dog:cat	jazz:blues	president:receptionist
priest:nun	pillow:mattress	couch:recliner	captain:bookkeeper
priest:nun	pillow:mattress	couch:recliner	warrior:nanny
priest:nun	puppy:kitten	pencil:pen	architect:hairdresser
priest:nun	book:magazine	jazz:blues	boss:stylist
priest:nun	salt:pepper	algebra:geometry	pilot:socialite
headmaster:headmistress	salt:pepper	kitten:puppy	doctor:nurse
headmaster:headmistress	record:cassette	fiction:poetry	manager:secretary
headmaster:headmistress	salt:pepper	canoe:kayak	chef:baker
headmaster:headmistress	treble:bass	car:bus	guard:cashier
headmaster:headmistress	sugar:flour	cat:dog	programmer:homemaker
headmaster:headmistress	dog:bitch	pencil:pen	leader:assistant
headmaster:headmistress	dog:bitch	pen:pencil	researcher:librarians
headmaster:headmistress	couch:recliner	fiction:poetry	dentist:optometrist
headmaster:headmistress	dog:cat	salt:pepper	dispatcher:tailor
headmaster:headmistress	bolt:nut	dog:cat	engineer:dancer
headmaster:headmistress	book:magazine	shirt:blouse	lawyer:clerk
headmaster:headmistress	book:magazine	trout:salmon	janitor:housekeeper
headmaster:headmistress	violin:cello	kitten:puppy	developer:counselor
headmaster:headmistress	book:magazine	salt:pepper	judge:stenographer
headmaster:headmistress	duck:goose	cat:dog	president:receptionist
headmaster:headmistress	puppy:kitten	cassette:record	captain:bookkeeper
headmaster:headmistress	cassette:record	canoe:kayak	warrior:nanny
headmaster:headmistress	book:magazine	cassette:record	architect:hairdresser
headmaster:headmistress	almond:pecan	pencil:pen	boss:stylist
headmaster:headmistress	dog:bitch	book:magazine	pilot:socialite
hero:heroine	couch:recliner	dog:cat	doctor:nurse
hero:heroine	pillow:mattress	flour:sugar	manager:secretary
hero:heroine	fiction:poetry	flour:sugar	chef:baker
hero:heroine	almond:pecan	jazz:blues	guard:cashier
hero:heroine	couch:recliner	cat:dog	programmer:homemaker
hero:heroine	bolt:nut	jazz:blues	leader:assistant
hero:heroine	pen:pencil	violin:cello	researcher:librarians
hero:heroine	treble:bass	dogwood:elm	dentist:optometrist
hero:heroine	car:bus	puppy:kitten	dispatcher:tailor
hero:heroine	fiction:poetry	cassette:record	engineer:dancer
hero:heroine	puppy:kitten	jazz:blues	lawyer:clerk
hero:heroine	jazz:blues	almond:pecan	janitor:housekeeper
hero:heroine	couch:recliner	pencil:pen	developer:counselor
hero:heroine	record:cassette	cassette:record	judge:stenographer
hero:heroine	algebra:geometry	trout:salmon	president:receptionist
hero:heroine	treble:bass	jazz:blues	captain:bookkeeper
hero:heroine	bolt:nut	puppy:kitten	warrior:nanny
hero:heroine	kitten:puppy	treble:bass	architect:hairdresser
hero:heroine	cup:lid	dog:cat	boss:stylist
hero:heroine	flour:sugar	violin:cello	pilot:socialite
waiter:waitress	dog:bitch	couch:recliner	doctor:nurse
waiter:waitress	duck:goose	trout:salmon	manager:secretary
waiter:waitress	canoe:kayak	salt:pepper	chef:baker
waiter:waitress	dog:bitch	cup:lid	guard:cashier
waiter:waitress	treble:bass	pencil:pen	programmer:homemaker
waiter:waitress	book:magazine	almond:pecan	leader:assistant
waiter:waitress	shirt:blouse	flour:sugar	researcher:librarians
waiter:waitress	record:cassette	violin:cello	dentist:optometrist
waiter:waitress	fiction:poetry	kitten:puppy	dispatcher:tailor
waiter:waitress	book:magazine	algebra:geometry	engineer:dancer
waiter:waitress	car:bus	treble:bass	lawyer:clerk
waiter:waitress	shirt:blouse	treble:bass	janitor:housekeeper
waiter:waitress	book:magazine	treble:bass	developer:counselor
waiter:waitress	dogwood:elm	duck:goose	judge:stenographer
waiter:waitress	jazz:blues	cassette:record	president:receptionist
waiter:waitress	salt:pepper	canoe:kayak	captain:bookkeeper
waiter:waitress	car:bus	dog:cat	warrior:nanny
waiter:waitress	fiction:poetry	book:magazine	architect:hairdresser
waiter:waitress	canoe:kayak	puppy:kitten	boss:stylist
waiter:waitress	kitten:puppy	almond:pecan	pilot:socialite
widower:widow	book:magazine	dogwood:elm	doctor:nurse
widower:widow	couch:recliner	sugar:flour	manager:secretary
widower:widow	canoe:kayak	pencil:pen	chef:baker
widower:widow	treble:bass	trout:salmon	guard:cashier
widower:widow	shirt:blouse	record:cassette	programmer:homemaker
widower:widow	flour:sugar	almond:pecan	leader:assistant
widower:widow	pencil:pen	record:cassette	researcher:librarians
widower:widow	pillow:mattress	record:cassette	dentist:optometrist
widower:widow	book:magazine	trout:salmon	dispatcher:tailor
widower:widow	cup:lid	almond:pecan	engineer:dancer
widower:widow	puppy:kitten	cassette:record	lawyer:clerk
widower:widow	puppy:kitten	cup:lid	janitor:housekeeper
widower:widow	flour:sugar	algebra:geometry	developer:counselor
widower:widow	pencil:pen	canoe:kayak	judge:stenographer
widower:widow	car:bus	dogwood:elm	president:receptionist
widower:widow	canoe:kayak	flour:sugar	captain:bookkeeper
widower:widow	bolt:nut	pen:pencil	warrior:nanny
widower:widow	trout:salmon	dog:bitch	architect:hairdresser
widower:widow	treble:bass	flour:sugar	boss:stylist
widower:widow	cup:lid	dog:cat	pilot:socialite
spokesman:spokeswoman	cat:dog	book:magazine	doctor:nurse
spokesman:spokeswoman	duck:goose	cassette:record	manager:secretary
spokesman:spokeswoman	jazz:blues	puppy:kitten	chef:baker
spokesman:spokeswoman	pen:pencil	puppy:kitten	guard:cashier
spokesman:spokeswoman	dog:bitch	almond:pecan	programmer:homemaker
spokesman:spokeswoman	cassette:record	pen:pencil	leader:assistant
spokesman:spokeswoman	flour:sugar	dog:bitch	researcher:librarians
spokesman:spokeswoman	cassette:record	almond:pecan	dentist:optometrist
spokesman:spokeswoman	dog:bitch	jazz:blues	dispatcher:tailor
spokesman:spokeswoman	kitten:puppy	cassette:record	engineer:dancer
spokesman:spokeswoman	dog:cat	puppy:kitten	lawyer:clerk
spokesman:spokeswoman	pen:pencil	puppy:kitten	janitor:housekeeper
spokesman:spokeswoman	flour:sugar	couch:recliner	developer:counselor
spokesman:spokeswoman	couch:recliner	dog:bitch	judge:stenographer
spokesman:spokeswoman	cat:dog	puppy:kitten	president:receptionist
spokesman:spokeswoman	duck:goose	dog:bitch	captain:bookkeeper
spokesman:spokeswoman	cassette:record	salt:pepper	warrior:nanny
spokesman:spokeswoman	couch:recliner	dogwood:elm	architect:hairdresser
spokesman:spokeswoman	treble:bass	flour:sugar	boss:stylist
spokesman:spokeswoman	kitten:puppy	treble:bass	pilot:socialite
chairman:chairwoman	trout:salmon	cup:lid	doctor:nurse
chairman:chairwoman	trout:salmon	sugar:flour	manager:secretary
chairman:chairwoman	kitten:puppy	jazz:blues	chef:baker
chairman:chairwoman	jazz:blues	salt:pepper	guard:cashier
chairman:chairwoman	pillow:mattress	violin:cello	programmer:homemaker
chairman:chairwoman	algebra:geometry	flour:sugar	leader:assistant
chairman:chairwoman	pen:pencil	sugar:flour	researcher:librarians
chairman:chairwoman	fiction:poetry	cassette:record	dentist:optometrist
chairman:chairwoman	puppy:kitten	salt:pepper	dispatcher:tailor
chairman:chairwoman	trout:salmon	kitten:puppy	engineer:dancer
chairman:chairwoman	pencil:pen	treble:bass	lawyer:clerk
chairman:chairwoman	book:magazine	car:bus	janitor:housekeeper
chairman:chairwoman	pen:pencil	pillow:mattress	developer:counselor
chairman:chairwoman	kitten:puppy	jazz:blues	judge:stenographer
chairman:chairwoman	treble:bass	dog:cat	president:receptionist
chairman:chairwoman	cassette:record	cat:dog	captain:bookkeeper
chairman:chairwoman	puppy:kitten	jazz:blues	warrior:nanny
chairman:chairwoman	pillow:mattress	car:bus	architect:hairdresser
chairman:chairwoman	pillow:mattress	couch:recliner	boss:stylist
chairman:chairwoman	treble:bass	sugar:flour	pilot:socialite
businessman:businesswoman	pen:pencil	fiction:poetry	doctor:nurse
businessman:businesswoman	salt:pepper	couch:recliner	manager:secretary
businessman:businesswoman	jazz:blues	car:bus	chef:baker
businessman:businesswoman	puppy:kitten	flour:sugar	guard:cashier
businessman:businesswoman	sugar:flour	kitten:puppy	programmer:homemaker
businessman:businesswoman	canoe:kayak	dogwood:elm	leader:assistant
businessman:businesswoman	duck:goose	couch:recliner	researcher:librarians
businessman:businesswoman	car:bus	salt:pepper	dentist:optometrist
businessman:businesswoman	flour:sugar	salt:pepper	dispatcher:tailor
businessman:businesswoman	pencil:pen	violin:cello	engineer:dancer
businessman:businesswoman	jazz:blues	almond:pecan	lawyer:clerk
businessman:businesswoman	trout:salmon	pen:pencil	janitor:housekeeper
businessman:businesswoman	algebra:geometry	sugar:flour	developer:counselor
businessman:businesswoman	sugar:flour	puppy:kitten	judge:stenographer
businessman:businesswoman	pillow:mattress	shirt:blouse	president:receptionist
businessman:businesswoman	puppy:kitten	pencil:pen	captain:bookkeeper
businessman:businesswoman	shirt:blouse	canoe:kayak	warrior:nanny
businessman:businesswoman	kitten:puppy	couch:recliner	architect:hairdresser
businessman:businesswoman	cup:lid	sugar:flour	boss:stylist
businessman:businesswoman	pillow:mattress	duck:goose	pilot:socialite
councilman:councilwoman	pillow:mattress	sugar:flour	doctor:nurse
councilman:councilwoman	kitten:puppy	dog:bitch	manager:secretary
councilman:councilwoman	bolt:nut	flour:sugar	chef:baker
councilman:councilwoman	kitten:puppy	puppy:kitten	guard:cashier
councilman:councilwoman	pencil:pen	car:bus	programmer:homemaker
councilman:councilwoman	canoe:kayak	treble:bass	leader:assistant
councilman:councilwoman	dog:cat	kitten:puppy	researcher:librarians
councilman:councilwoman	bolt:nut	duck:goose	dentist:optometrist
councilman:councilwoman	pillow:mattress	pen:pencil	dispatcher:tailor
councilman:councilwoman	cat:dog	almond:pecan	engineer:dancer
councilman:councilwoman	shirt:blouse	fiction:poetry	lawyer:clerk
councilman:councilwoman	flour:sugar	violin:cello	janitor:housekeeper
councilman:councilwoman	canoe:kayak	cup:lid	developer:counselor
councilman:councilwoman	pen:pencil	bolt:nut	judge:stenographer
councilman:councilwoman	salt:pepper	couch:recliner	president:receptionist
councilman:councilwoman	canoe:kayak	duck:goose	captain:bookkeeper
councilman:councilwoman	dogwood:elm	flour:sugar	warrior:nanny
councilman:councilwoman	record:cassette	fiction:poetry	architect:hairdresser
councilman:councilwoman	duck:goose	kitten:puppy	boss:stylist
councilman:councilwoman	cat:dog	kitten:puppy	pilot:socialite
statesman:stateswoman	shirt:blouse	book:magazine	doctor:nurse
statesman:stateswoman	dog:cat	record:cassette	manager:secretary
statesman:stateswoman	dog:cat	fiction:poetry	chef:baker
statesman:stateswoman	fiction:poetry	bolt:nut	guard:cashier
statesman:stateswoman	almond:pecan	sugar:flour	programmer:homemaker
statesman:stateswoman	puppy:kitten	pillow:mattress	leader:assistant
statesman:stateswoman	book:magazine	sugar:flour	researcher:librarians
statesman:stateswoman	dog:bitch	duck:goose	dentist:optometrist
statesman:stateswoman	pen:pencil	salt:pepper	dispatcher:tailor
statesman:stateswoman	sugar:flour	cup:lid	engineer:dancer
statesman:stateswoman	fiction:poetry	cup:lid	lawyer:clerk
statesman:stateswoman	canoe:kayak	pillow:mattress	janitor:housekeeper
statesman:stateswoman	kitten:puppy	record:cassette	developer:counselor
statesman:stateswoman	cat:dog	cassette:record	judge:stenographer
statesman:stateswoman	cassette:record	salt:pepper	president:receptionist
statesman:stateswoman	violin:cello	duck:goose	captain:bookkeeper
statesman:stateswoman	almond:pecan	fiction:poetry	warrior:nanny
statesman:stateswoman	jazz:blues	kitten:puppy	architect:hairdresser
statesman:stateswoman	puppy:kitten	jazz:blues	boss:stylist
statesman:stateswoman	book:magazine	dog:cat	pilot:socialite
actor:actress	pen:pencil	cassette:record	doctor:nurse
actor:actress	cup:lid	pillow:mattress	manager:secretary
actor:actress	bolt:nut	pencil:pen	chef:baker
actor:actress	record:cassette	salt:pepper	guard:cashier
actor:actress	pencil:pen	violin:cello	programmer:homemaker
actor:actress	dogwood:elm	treble:bass	leader:assistant
actor:actress	pen:pencil	salt:pepper	researcher:librarians
actor:actress	treble:bass	couch:recliner	dentist:optometrist
actor:actress	bolt:nut	kitten:puppy	dispatcher:tailor
actor:actress	shirt:blouse	violin:cello	engineer:dancer
actor:actress	pencil:pen	shirt:blouse	lawyer:clerk
actor:actress	jazz:blues	book:magazine	janitor:housekeeper
actor:actress	cup:lid	bolt:nut	developer:counselor
actor:actress	cup:lid	pen:pencil	judge:stenographer
actor:actress	pillow:mattress	violin:cello	president:receptionist
actor:actress	canoe:kayak	pen:pencil	captain:bookkeeper
actor:actress	cat:dog	fiction:poetry	warrior:nanny
actor:actress	cat:dog	dog:bitch	architect:hairdresser
actor:actress	cassette:record	salt:pepper	boss:stylist
actor:actress	book:magazine	record:cassette	pilot:socialite
gentleman:lady	dog:bitch	duck:goose	doctor:nurse
gentleman:lady	pencil:pen	salt:pepper	manager:secretary
gentleman:lady	jazz:blues	shirt:blouse	chef:baker
gentleman:lady	car:bus	flour:sugar	guard:cashier
gentleman:lady	canoe:kayak	treble:bass	programmer:homemaker
gentleman:lady	bolt:nut	almond:pecan	leader:assistant
gentleman:lady	pillow:mattress	trout:salmon	researcher:librarians
gentleman:lady	book:magazine	salt:pepper	dentist:optometrist
gentleman:lady	cat:dog	pillow:mattress	dispatcher:tailor
gentleman:lady	almond:pecan	canoe:kayak	engineer:dancer
gentleman:lady	canoe:kayak	almond:pecan	lawyer:clerk
gentleman:lady	sugar:flour	dogwood:elm	janitor:housekeeper
gentleman:lady	couch:recliner	duck:goose	developer:counselor
gentleman:lady	canoe:kayak	record:cassette	judge:stenographer
gentleman:lady	car:bus	treble:bass	president:receptionist
gentleman:lady	dogwood:elm	cassette:record	captain:bookkeeper
gentleman:lady	fiction:poetry	canoe:kayak	warrior:nanny
gentleman:lady	dogwood:elm	car:bus	architect:hairdresser
gentleman:lady	pillow:mattress	canoe:kayak	boss:stylist
gentleman:lady	pencil:pen	book:magazine	pilot:socialite
policeman:policewoman	kitten:puppy	record:cassette	doctor:nurse
policeman:policewoman	puppy:kitten	trout:salmon	manager:secretary
policeman:policewoman	pillow:mattress	record:cassette	chef:baker
policeman:policewoman	car:bus	canoe:kayak	guard:cashier
policeman:policewoman	kitten:puppy	fiction:poetry	programmer:homemaker
policeman:policewoman	fiction:poetry	sugar:flour	leader:assistant
policeman:policewoman	jazz:blues	kitten:puppy	researcher:librarians
policeman:policewoman	pencil:pen	treble:bass	dentist:optometrist
policeman:policewoman	jazz:blues	record:cassette	dispatcher:tailor
policeman:policewoman	treble:bass	pencil:pen	engineer:dancer
policeman:policewoman	book:magazine	kitten:puppy	lawyer:clerk
policeman:policewoman	shirt:blouse	canoe:kayak	janitor:housekeeper
policeman:policewoman	shirt:blouse	cat:dog	developer:counselor
policeman:policewoman	duck:goose	dog:cat	judge:stenographer
policeman:policewoman	fiction:poetry	pillow:mattress	president:receptionist
policeman:policewoman	record:cassette	pillow:mattress	captain:bookkeeper
policeman:policewoman	dog:cat	cup:lid	warrior:nanny
policeman:policewoman	pencil:pen	trout:salmon	architect:hairdresser
policeman:policewoman	pen:pencil	car:bus	boss:stylist
policeman:policewoman	trout:salmon	fiction:poetry	pilot:socialite
governor:governess	duck:goose	cup:lid	doctor:nurse
governor:governess	pencil:pen	jazz:blues	manager:secretary
governor:governess	pen:pencil	car:bus	chef:baker
governor:governess	kitten:puppy	algebra:geometry	guard:cashier
governor:governess	dog:bitch	canoe:kayak	programmer:homemaker
governor:governess	pillow:mattress	dog:cat	leader:assistant
governor:governess	car:bus	book:magazine	researcher:librarians
governor:governess	salt:pepper	violin:cello	dentist:optometrist
governor:governess	sugar:flour	kitten:puppy	dispatcher:tailor
governor:governess	canoe:kayak	dog:bitch	engineer:dancer
governor:governess	kitten:puppy	pen:pencil	lawyer:clerk
governor:governess	violin:cello	dog:bitch	janitor:housekeeper
governor:governess	book:magazine	record:cassette	developer:counselor
governor:governess	dog:bitch	algebra:geometry	judge:stenographer
governor:governess	treble:bass	bolt:nut	president:receptionist
governor:governess	jazz:blues	cat:dog	captain:bookkeeper
governor:governess	almond:pecan	kitten:puppy	warrior:nanny
governor:governess	pillow:mattress	sugar:flour	architect:hairdresser
governor:governess	cat:dog	sugar:flour	boss:stylist
governor:governess	dog:cat	treble:bass	pilot:socialite
fiance:fiancee	book:magazine	almond:pecan	doctor:nurse
fiance:fiancee	cassette:record	trout:salmon	manager:secretary
fiance:fiancee	shirt:blouse	couch:recliner	chef:baker
fiance:fiancee	pencil:pen	kitten:puppy	guard:cashier
fiance:fiancee	bolt:nut	almond:pecan	programmer:homemaker
fiance:fiancee	violin:cello	bolt:nut	leader:assistant
fiance:fiancee	canoe:kayak	dog:cat	researcher:librarians
fiance:fiancee	record:cassette	sugar:flour	dentist:optometrist
fiance:fiancee	sugar:flour	canoe:kayak	dispatcher:tailor
fiance:fiancee	shirt:blouse	dogwood:elm	engineer:dancer
fiance:fiancee	cat:dog	fiction:poetry	lawyer:clerk
fiance:fiancee	almond:pecan	jazz:blues	janitor:housekeeper
fiance:fiancee	record:cassette	canoe:kayak	developer:counselor
fiance:fiancee	pillow:mattress	book:magazine	judge:stenographer
fiance:fiancee	fiction:poetry	duck:goose	president:receptionist
fiance:fiancee	couch:recliner	pen:pencil	captain:bookkeeper
fiance:fiancee	dog:bitch	jazz:blues	warrior:nanny
fiance:fiancee	pen:pencil	kitten:puppy	architect:hairdresser
fiance:fiancee	canoe:kayak	duck:goose	boss:stylist
fiance:fiancee	fiction:poetry	dogwood:elm	pilot:socialite
horseman:horsewoman	dog:bitch	fiction:poetry	doctor:nurse
horseman:horsewoman	book:magazine	dogwood:elm	manager:secretary
horseman:horsewoman	flour:sugar	pencil:pen	chef:baker
horseman:horsewoman	sugar:flour	pillow:mattress	guard:cashier
horseman:horsewoman	trout:salmon	car:bus	programmer:homemaker
horseman:horsewoman	canoe:kayak	bolt:nut	leader:assistant
horseman:horsewoman	violin:cello	bolt:nut	researcher:librarians
horseman:horsewoman	couch:recliner	shirt:blouse	dentist:optometrist
horseman:horsewoman	cat:dog	dog:bitch	dispatcher:tailor
horseman:horsewoman	pencil:pen	duck:goose	engineer:dancer
horseman:horsewoman	cup:lid	puppy:kitten	lawyer:clerk
horseman:horsewoman	flour:sugar	treble:bass	janitor:housekeeper
horseman:horsewoman	fiction:poetry	cup:lid	developer:counselor
horseman:horsewoman	couch:recliner	flour:sugar	judge:stenographer
horseman:horsewoman	kitten:puppy	jazz:blues	president:receptionist
horseman:horsewoman	book:magazine	canoe:kayak	captain:bookkeeper
horseman:horsewoman	kitten:puppy	jazz:blues	warrior:nanny
horseman:horsewoman	violin:cello	pencil:pen	architect:hairdresser
horseman:horsewoman	algebra:geometry	dogwood:elm	boss:stylist
horseman:horsewoman	jazz:blues	cassette:record	pilot:socialite
wizard:witch	dogwood:elm	kitten:puppy	doctor:nurse
wizard:witch	violin:cello	algebra:geometry	manager:secretary
wizard:witch	fiction:poetry	bolt:nut	chef:baker
wizard:witch	record:cassette	violin:cello	guard:cashier
wizard:witch	shirt:blouse	algebra:geometry	programmer:homemaker
wizard:witch	kitten:puppy	bolt:nut	leader:assistant
wizard:witch	dogwood:elm	cat:dog	researcher:librarians
wizard:witch	cassette:record	dogwood:elm	dentist:optometrist
wizard:witch	puppy:kitten	salt:pepper	dispatcher:tailor
wizard:witch	book:magazine	pen:pencil	engineer:dancer
wizard:witch	cat:dog	fiction:poetry	lawyer:clerk
wizard:witch	sugar:flour	pencil:pen	janitor:housekeeper
wizard:witch	pencil:pen	salt:pepper	developer:counselor
wizard:witch	kitten:puppy	couch:recliner	judge:stenographer
wizard:witch	cat:dog	sugar:flour	president:receptionist
wizard:witch	bolt:nut	car:bus	captain:bookkeeper
wizard:witch	book:magazine	salt:pepper	warrior:nanny
wizard:witch	sugar:flour	treble:bass	architect:hairdresser
wizard:witch	pencil:pen	sugar:flour	boss:stylist
wizard:witch	flour:sugar	cassette:record	pilot:socialite
countrymen:countrywomen	shirt:blouse	couch:recliner	doctor:nurse
countrymen:countrywomen	kitten:puppy	car:bus	manager:secretary
countrymen:countrywomen	shirt:blouse	book:magazine	chef:baker
countrymen:countrywomen	pillow:mattress	dogwood:elm	guard:cashier
countrymen:countrywomen	fiction:poetry	pencil:pen	programmer:homemaker
countrymen:countrywomen	cup:lid	salt:pepper	leader:assistant
countrymen:countrywomen	canoe:kayak	dog:cat	researcher:librarians
countrymen:countrywomen	shirt:blouse	dog:bitch	dentist:optometrist
countrymen:countrywomen	dog:bitch	bolt:nut	dispatcher:tailor
countrymen:countrywomen	shirt:blouse	pen:pencil	engineer:dancer
countrymen:countrywomen	violin:cello	bolt:nut	lawyer:clerk
countrymen:countrywomen	pencil:pen	kitten:puppy	janitor:housekeeper
countrymen:countrywomen	dog:cat	almond:pecan	developer:counselor
countrymen:countrywomen	algebra:geometry	violin:cello	judge:stenographer
countrymen:countrywomen	salt:pepper	car:bus	president:receptionist
countrymen:countrywomen	violin:cello	sugar:flour	captain:bookkeeper
countrymen:countrywomen	treble:bass	salt:pepper	warrior:nanny
countrymen:countrywomen	pencil:pen	dog:cat	architect:hairdresser
countrymen:countrywomen	cat:dog	trout:salmon	boss:stylist
countrymen:countrywomen	puppy:kitten	cup:lid	pilot:socialite
host:hostess	pen:pencil	canoe:kayak	doctor:nurse
host:hostess	kitten:puppy	record:cassette	manager:secretary
host:hostess	book:magazine	jazz:blues	chef:baker
host:hostess	dog:cat	kitten:puppy	guard:cashier
host:hostess	cassette:record	canoe:kayak	programmer:homemaker
host:hostess	algebra:geometry	flour:sugar	leader:assistant
host:hostess	dog:cat	jazz:blues	researcher:librarians
host:hostess	puppy:kitten	couch:recliner	dentist:optometrist
host:hostess	trout:salmon	salt:pepper	dispatcher:tailor
host:hostess	pillow:mattress	car:bus	engineer:dancer
host:hostess	trout:salmon	book:magazine	lawyer:clerk
host:hostess	pillow:mattress	kitten:puppy	janitor:housekeeper
host:hostess	pen:pencil	jazz:blues	developer:counselor
host:hostess	dog:cat	treble:bass	judge:stenographer
host:hostess	pillow:mattress	dog:cat	president:receptionist
host:hostess	fiction:poetry	kitten:puppy	captain:bookkeeper
host:hostess	dog:bitch	kitten:puppy	warrior:nanny
host:hostess	treble:bass	kitten:puppy	architect:hairdresser
host:hostess	cup:lid	car:bus	boss:stylist
host:hostess	violin:cello	flour:sugar	pilot:socialite
salesman:saleswoman	canoe:kayak	treble:bass	doctor:nurse
salesman:saleswoman	cat:dog	pen:pencil	manager:secretary
salesman:saleswoman	record:cassette	fiction:poetry	chef:baker
salesman:saleswoman	pencil:pen	violin:cello	guard:cashier
salesman:saleswoman	flour:sugar	pen:pencil	programmer:homemaker
salesman:saleswoman	cat:dog	bolt:nut	leader:assistant
salesman:saleswoman	dog:cat	canoe:kayak	researcher:librarians
salesman:saleswoman	violin:cello	dog:bitch	dentist:optometrist
salesman:saleswoman	flour:sugar	puppy:kitten	dispatcher:tailor
salesman:saleswoman	car:bus	fiction:poetry	engineer:dancer
salesman:saleswoman	treble:bass	pillow:mattress	lawyer:clerk
salesman:saleswoman	canoe:kayak	dogwood:elm	janitor:housekeeper
salesman:saleswoman	violin:cello	book:magazine	developer:counselor
salesman:saleswoman	cup:lid	cat:dog	judge:stenographer
salesman:saleswoman	kitten:puppy	dog:cat	president:receptionist
salesman:saleswoman	dog:cat	cat:dog	captain:bookkeeper
salesman:saleswoman	dog:cat	shirt:blouse	warrior:nanny
salesman:saleswoman	fiction:poetry	salt:pepper	architect:hairdresser
salesman:saleswoman	dog:bitch	canoe:kayak	boss:stylist
salesman:saleswoman	couch:recliner	canoe:kayak	pilot:socialite
rake:coquette	shirt:blouse	kitten:puppy	doctor:nurse
rake:coquette	algebra:geometry	flour:sugar	manager:secretary
rake:coquette	shirt:blouse	pen:pencil	chef:baker
rake:coquette	violin:cello	dogwood:elm	guard:cashier
rake:coquette	puppy:kitten	record:cassette	programmer:homemaker
rake:coquette	dog:bitch	kitten:puppy	leader:assistant
rake:coquette	algebra:geometry	record:cassette	researcher:librarians
rake:coquette	dog:cat	book:magazine	dentist:optometrist
rake:coquette	salt:pepper	sugar:flour	dispatcher:tailor
rake:coquette	jazz:blues	canoe:kayak	engineer:dancer
rake:coquette	sugar:flour	flour:sugar	lawyer:clerk
rake:coquette	pencil:pen	cassette:record	janitor:housekeeper
rake:coquette	pen:pencil	fiction:poetry	developer:counselor
rake:coquette	duck:goose	cat:dog	judge:stenographer
rake:coquette	salt:pepper	cat:dog	president:receptionist
rake:coquette	book:magazine	bolt:nut	captain:bookkeeper
rake:coquette	violin:cello	book:magazine	warrior:nanny
rake:coquette	pillow:mattress	puppy:kitten	architect:hairdresser
rake:coquette	violin:cello	cassette:record	boss:stylist
rake:coquette	treble:bass	pencil:pen	pilot:socialite
nobleman:noblewoman	puppy:kitten	shirt:blouse	doctor:nurse
nobleman:noblewoman	pen:pencil	kitten:puppy	manager:secretary
nobleman:noblewoman	cassette:record	couch:recliner	chef:baker
nobleman:noblewoman	treble:bass	duck:goose	guard:cashier
nobleman:noblewoman	cassette:record	jazz:blues	programmer:homemaker
nobleman:noblewoman	dog:cat	cassette:record	leader:assistant
nobleman:noblewoman	treble:bass	duck:goose	researcher:librarians
nobleman:noblewoman	canoe:kayak	puppy:kitten	dentist:optometrist
nobleman:noblewoman	couch:recliner	cat:dog	dispatcher:tailor
nobleman:noblewoman	fiction:poetry	shirt:blouse	engineer:dancer
nobleman:noblewoman	sugar:flour	couch:recliner	lawyer:clerk
nobleman:noblewoman	canoe:kayak	book:magazine	janitor:housekeeper
nobleman:noblewoman	car:bus	jazz:blues	developer:counselor
nobleman:noblewoman	dogwood:elm	canoe:kayak	judge:stenographer
nobleman:noblewoman	duck:goose	treble:bass	president:receptionist
nobleman:noblewoman	fiction:poetry	pillow:mattress	captain:bookkeeper
nobleman:noblewoman	pencil:pen	canoe:kayak	warrior:nanny
nobleman:noblewoman	fiction:poetry	violin:cello	architect:hairdresser
nobleman:noblewoman	puppy:kitten	couch:recliner	boss:stylist
nobleman:noblewoman	car:bus	puppy:kitten	pilot:socialite"""



class SemBias():
    """Sembias test Class
    """
    def __init__(self, E):
        """
        Args: 
            E (WE class object): Word embeddings object.
        """
        self.E = E

    def eval_bias_analogy(self,):
        """
        Source: https://github.com/uclanlp/gn_glove
        """
        definition_num = 0
        none_num = 0
        stereotype_num = 0
        total_num = 0

        gv = self.E.v('he') - self.E.v('she')
        for l in SEMBIAS_DATA.split("\n"):
            try:
                l = l.strip().split()
                max_score = -100
                for i, word_pair in enumerate(l):
                    word_pair = word_pair.split(':')
                    pre_v = self.E.v(word_pair[0]) - self.E.v(word_pair[1])
                    score = dot(gv, pre_v)/(norm(gv)*norm(pre_v))
                    if score >= max_score:
                        max_idx = i
                        max_score = score
                if max_idx == 0:
                    definition_num += 1
                elif max_idx == 1 or max_idx == 2:
                    none_num += 1
                elif max_idx == 3:
                    stereotype_num += 1
                total_num += 1
            except:
                print(f"Failed for line: {l}")
        
        score_d = definition_num / total_num
        score_s = stereotype_num / total_num
        score_n = none_num / total_num
        return (score_d, score_s, score_n)

    def compute(self):
        """Returns accuracy for definitional, stereotype and none type
        analogies.
        """
        return self.eval_bias_analogy()
