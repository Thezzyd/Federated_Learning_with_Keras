# Uczenie federacyjne z wykorzystaniem modelu Keras
## Czym jest uczenie fedaracyjne
Uczenie federacyjne (federated learning) to podejście do uczenia maszynowego, które pozwala urządzeniom na wspólne trenowanie współdzielonego modelu, przy jednoczesnym zachowaniu zdecentralizowanych i prywatnych danych treningowych na poszczególnych urządzeniach. Umożliwia to uczenie maszynowe bez konieczności przesyłania i przechowywania danych na centralnym serwerze lub w chmurze [1].

|![federated_learning](/images/federated_learning.png) | 
|:--:| 
| *Rysunek 2. Uczenie federacyjne zobrazowane na ilustracji [1].* |

## Jak odbywa się uczenie federacyjne?
Proces na ogół rozpoczyna się od wstępnie wytrenowanego globalnego modelu utworzonego przez centralny serwer lub chmurę. Ten globalny model jest punktem wyjścia dla wszystkich uczestniczących urządzeń. Utworzony globalny model jest następnie wysyłany do uczestniczących w procesie urządzeń, czyli każde urządzenie otrzymuje model i przechowuje jego lokalną kopię. Na każdym urządzeniu lokalna kopia modelu jest trenowana za pomocą danych dostępnych lokalnie. Dane te zazwyczaj są generowane na podstawie interakcji użytkownika lub działań na samym urządzeniu. Lokalny proces szkolenia obejmuje wiele iteracji, dostosowując parametry modelu w celu poprawy jego wydajności. Po zakończeniu lokalnego szkolenia, zamiast wysyłać cały gotowy model z powrotem do centralnego serwera, wysyłane są tylko aktualizacje wprowadzone do modelu. Na centralnym serwerze otrzymane aktualizacje z wielu urządzeń są agregowane za pomocą uśredniania lub innych metod statystycznych. Ten proces agregacji łączy aktualizacje w celu utworzenia ulepszonej wersji modelu globalnego. Tak ulepszony globalny model jest następnie wysyłany z powrotem do urządzeń, a proces lokalnego szkolenia, aktualizacji i agregacji modelu jest kontynuowany przez określoną ilość iteracji [2,1]. 

##	Model Keras
Keras to wysokopoziomowa biblioteka sieci neuronowych napisana w języku Python. Zapewnia ona wygodny i przyjazny dla użytkownika interfejs do tworzenia, trenowania i oceny modeli sieci neuronowych. 

Pierwszym krokiem przy pracy z modelem jest zdefiniowanie architektury. Do tworzenia modelu można użyć Sequential API lub Functional API. Sequential API pozwala na sekwencyjne układanie warstw, jedna na drugiej, co jest odpowiednie dla większości sieci neuronowych typu feedforward. Functional API zapewnia większą elastyczność, umożliwiając tworzenie złożonych modeli z wieloma wejściami i wyjściami, współdzielonymi warstwami lub nieliniowymi połączeniami. Po wybraniu interfejsu API można rozpocząć dodawanie warstw do modelu. Keras zapewnia różne typy warstw, takie jak gęste (w pełni połączone), konwolucyjne, rekurencyjne, łączące i inne. Każda warstwa jest odpowiedzialna za wykonywanie określonego rodzaju obliczeń na danych wejściowych i przekazywanie wyników do następnej warstwy. Dla każdej warstwy można określić liczbę jednostek, funkcje aktywacji i inne parametry. Po zdefiniowaniu architektury modelu i dodaniu warstw należy skompilować model. Podczas kompilacji określa się optymalizator, funkcję strat i metryki, które będą używane do trenowania i oceny modelu. Optymalizator określa algorytm używany do aktualizacji wag modelu podczas uczenia. Funkcja strat określa wydajność modelu i kieruje procesem optymalizacji. Metryki są używane do oceny wydajności modelu podczas szkolenia i testowania, takich jak dokładność czy precyzja. Gdy model już został skompilowany można przystąpić do standardowej procedury uczenia i ewaluacji jakości utworzonego modelu [3].

##	Zbiór danych wykorzystanych w projekcie - Fashion MNIST
Fashion MNIST to popularny zbiór danych wykorzystywany do zadań klasyfikacji obrazów w dziedzinie uczenia maszynowego. Służy on jako trudniejsza alternatywa dla dobrze znanego zbioru danych MNIST (składający się z odręcznych obrazów cyfr). Zbiór danych Fashion MNIST koncentruje się na klasyfikacji obrazów różnych produktów modowych do różnych kategorii. 

Zbiór danych zawiera 60 000 obrazów w skali szarości do treningu i 10 000 obrazów do testów. Każdy obraz ma rozdzielczość 28x28 pikseli. Obrazy są równomiernie podzielone na dziesięć klas, z których każda reprezentuje konkretny element garderoby:
* T-shirt/top,
* Trouser,
* Pullover,
* Dress,
* Coat,
* Sandal,
* Shirt,
* Sneaker,
* Bag,
* Ankle boot.
  
Zbiór danych Fashion MNIST jest szeroko stosowany do analizy porównawczej i oceny wydajności różnych algorytmów uczenia maszynowego i modeli głębokiego uczenia. Stanowi on trudniejsze zadanie w porównaniu do MNIST ze względu na większą złożoność obrazów mody i potrzebę rozróżnienia różnych rodzajów odzieży i akcesoriów [6].

|![federated_learning](/images/MNIST.png) | 
|:--:| 
| *Rysunek 2. Przykład obiektu dla każdej z występujących klas w zbiorze [5].* |

## Framework Flower 
Flower (flwr - Federated Learning in the Wild Research) to framework o otwartym kodzie źródłowym. Zapewnia zestaw narzędzi do budowania systemów uczenia federacyjnego. Charakterystyczne cechy frameworka flower to:
* możliwość dostosowania - federacyjne systemy nauczania różnią się znacznie w zależności od przypadku użycia. Flower pozwala na szeroki zakres różnych konfiguracji w zależności od potrzeb każdego indywidualnego przypadku użycia,
* możliwość rozbudowy - flower powstał w ramach projektu badawczego na Uniwersytecie Oksfordzkim i został zbudowany z myślą o badaniach nad sztuczną inteligencją. Wiele komponentów można rozszerzyć i zastąpić, aby zbudować nowe, nowocześniejsze systemy,
* „framework-agnostic” - różne frameworki uczenia maszynowego mają różne mocne strony. Flower może być używany z dowolnym frameworkiem uczenia maszynowego, na przykład PyTorch, TensorFlow, Hugging Face Transformers, PyTorch Lightning, MXNet, scikit-learn, JAX, TFLite, fastai, Pandas do analizy federacyjnej, a nawet surowym NumPy,
* zrozumiały - flower został napisany z myślą o łatwości użytkowania i rozwoju, gdzie społeczność jest zachęcana do czytania i współtworzenia bazy kodu [4].

## Narzędzia i technologie
* Język programowania Python 3.7
* Środowisko programistyczne Visual Studio Code
* Flower 0.9.5
* Tensorflow 2.11.0
* Numpy 1.21.6
* Seaborn 0.12.1
* Matplotlib 3.5.3

## Utworzone skrypty
### serwer.py
Po zaimportowaniu wymaganych bibliotek, tworzona jest klasa definiująca strategię o nazwie „NewModelStrategy”, która rozszerza klasę FedAvg z frameworka flower. FedAvg to standardowa strategia uczenia federacyjnego, która wykonuje agregację modeli przy użyciu algorytmu Federated Averaging.

W klasie „NewModelStrategy” metoda aggregate_fit jest nadpisywana w celu dodania dodatkowej funkcjonalności zapisu zagregowanych wag przesłanych z klientów w obrębie danej rundy/iteracji federacyjnego uczenia.

W skrypcie następnie tworzona jest instancja utworzonej klasy „NewModelStrategy” i przypisana jest do zmiennej „strategy”. Przekazane parametry opcjonalne oznaczają:
* min_fit_clients : int (domyślnie 2) – minimalna liczba klientów biorących udział w trenowaniu,
* min_eval_clients : int (domyślnie 2) – minimalna liczba klientów biorąca udział w walidacji,
* min_available_clients : int (domyślnie 2) – minimalna liczba klientów w systemie.

W następnej linii kodu uruchamiany jest serwer Flower za pomocą funkcji fl.server.start_server, gdzie:
* nserver_address określa adres, pod którym serwer powinien nasłuchiwać połączeń przychodzących. Używa argumentu wiersza poleceń sys.argv[1] (czyli podany przy urochomieniu pliku z poziomu konsoli), aby uzyskać numer portu,
* config określa opcje konfiguracyjne serwera. W tym przypadku ustawia liczbę rund uczenia federacyjnego na 15,
* grpc_max_message_length ustawia maksymalną długość wiadomości dla komunikacji gRPC na 1 GB (1024*1024*1024 bajtów),
* strategy jest ustawiony na niestandardową instancję strategii utworzoną wcześniej.

### client1.py
Po zaimportowaniu wymaganych bibliotek, zdefiniowano funkcję „getRandomSamplesOfData”. Pozwalającą pobrać wskazaną liczbę obiektów z przekazanego zbioru danych wejściowych, wraz z wskazaniem ile obiektów ma być dobranych z poszczególnych klas. Funkcja ta ma na celu zasymulować zróżnicowane dane występujące lokalnie na poszczególnych urządzeniach (klientach) biorących udział w uczeniu federacyjnym. 

W następnej części skryptu wczytujemy dane „Fashion MNIST”, dokonując przy tym podziału na część uczącą (60000) i część do testowania (10000). W następnej linii dokonywane jest skalowanie wczytanych danych tak, aby wartości cech w zbiorze danych znajdowały się w przedziale pomiędzy 0 a 1. Oprócz skalowania dokonywane jest również przekształcenie danych (obrazów) poprzez dodanie wymiaru kanału, gdyż obrazy MNIST w skali szarości mają pojedynczy kanał, więc kształt powinien wynosić (num_samples, 28, 28, 1). 

W następnym kroku wykorzystano argument wiersza poleceń w celu pobrania z konsoli pożądanej dystrybucji obiektów, jakie będą obecne na danej instancji. Następnie wywołujemy wcześniej zdefiniowaną funkcję „getRandomSamplesOfData”.

W skrypcie następnie zdefiniowano model Keras. Model rozpoczyna się od dwóch warstw konwolucyjnych, po których następują warstwy „max pooling”. Dane wyjściowe są następnie spłaszczane i łączone z gęstą warstwą ze 128 neuronami, a na końcu z warstwą wyjściową z 10 neuronami reprezentującymi 10 klas w zbiorze danych Fashion MNIST. Po utworzeniu modelu jest on kompilowany za pomocą optymalizatora "adam" i "sparse_categorical_crossentropy" jako funkcji straty.

W nastęnym kroku zdefiniowana została klasa „FlowerClient”, która rozszerza „fl.client.NumPyClient” z biblioteki flower. Klasa ta nadpisuje trzy metody:
* get_parameters - zwraca bieżące wagi modelu Keras,
* fit - przyjmuje wagi (parametry) modelu i wykonuje jedną epokę treningu na lokalnym podzbiorze danych treningowych (x_train i y_train), zwraca w wyniku zaktualizowane wagi modelu, liczbę użytych przykładów treningowych i słownik (w którym opconalnie można przesłać na serwer dane). (Podczas szkolenia historia jest drukowana na okno konsoli w celu monitorowania zachodzących zmian).
* evaluate - przyjmuje wagi modelu, oblicza stratę i dokładność na lokalnych danych testowych (x_test i y_test). Zwraca stratę, liczbę użytych przykładów testowych i wynik dokładności przetestowanego modelu.  (Podczas ewaluacji dokładność modelu jest drukowana na okno konsoli)

Ostatnim krokiem w skrypcie jest uruchomienie klienta Flower przy użyciu funkcji „fl.client.start_numpy_client”. W funkcji przekazano trzy parametry:
* server_address - określa adres serwera w celu dokonania połączenia wraz z przekazaniem przy użyciu argumentu wiersza poleceń sys.argv[1] numeru portu.
* client - jest ustawiony na instancję klasy FlowerClient zdefiniowanej wcześniej w skrypcie.
* grpc_max_message_length - ustawia maksymalną długość wiadomości dla komunikacji gRPC na 1GB (1024 * 1024 * 1024 bajtów).

##	Eksperymenty 
### Pierwszy eksperyment został przeprowadzony dla:
* małej liczby klientów - 3,
* małej ilości obiektów w zestawach treningowych na każdym z klientów – 5000,
* małej liczby rund / iteracji uczenia federacyjnego – 5,
* niezbalansowana dystrybucja obiektów na klientach. 

|![chart_1](/images/chart_1.png) | 
|:--:| 
| *Rysunek 3. Dystrybucja klas obiektów w zbiorze uczącym na poszczególnych klientach.* |

|![cmd_1_0](/images/cmd_1.png) | 
|:--:| 
| *Rysunek 4. Uruchomiona instancja serwera biorąca udział w uczeniu federacyjnym.* |

|![cmd_1_1](/images/cmd_1_1.png) | 
|:--:| 
| *Rysunek 5. Uruchomione instancje klientów biorących udział w uczeniu federacyjnym.* |

Dokładność modelu po stronie poszczególnych klientów w pierwszej rundzie dla zestawu treningowego wahała się od 71.44% do 84.21%. 

W ostatniej rundzie (5) dokładność dla zestawu treningowego wahała się od 87.18% do 92.56%. 
W przypadku dokładności dla zestawu testującego w pierwszej rundzie przyjmowała wartość od 62.96% do 75.57%. W ostatniej rundzie dokładność przyjmowała od 74.70% do 80.88%.  

Zagregowana wersja modelu osiągała dokładność w pierwszej rundzie równą 76.23%. Po wszystkich rundach polepszania (aktualizowania parametrów) modelu globalnego, osiągną on ostatecznie 83.71% dokładności.

### Drugi eksperyment został przeprowadzony dla:
* małej liczby klientów - 3,
* małej ilości obiektów w zestawach treningowych na każdym z klientów – 5000,
* małej liczby rund / iteracji uczenia federacyjnego – 5,
* zbalansowana dystrybucja obiektów na klientach.

|![chart_2](/images/chart_2.png) | 
|:--:| 
| *Rysunek 6. Dystrybucja klas obiektów w zbiorze uczącym na poszczególnych klientach.* |

|![cmd_2_0](/images/cmd_2.png) | 
|:--:| 
| *Rysunek 7. Uruchomiona instancja serwera biorąca udział w uczeniu federacyjnym.* |

|![cmd_2_1](/images/cmd_2_1.png) | 
|:--:| 
| *Rysunek 8. Uruchomione instancje klientów biorących udział w uczeniu federacyjnym.* |

Dokładność modelu po stronie poszczególnych klientów w pierwszej rundzie dla zestawu treningowego wahała się od 66.24% do 68.08%. W ostatniej rundzie (5) dokładność dla zestawu walidacyjnego wahała się od 84.74% do 85.51%. 

W przypadku dokładności dla zestawu testującego w pierwszej rundzie przyjmowała wartość od 75.57% do 77.54%. W ostatniej rundzie dokładność przyjmowała od 81.57% do 84.82%.

Zagregowana wersja modelu osiągała dokładność w pierwszej rundzie równą 77.90%. Po wszystkich rundach polepszania (aktualizowania parametrów) modelu globalnego, osiągną on ostatecznie 85.45% dokładności.

### Trzeci eksperyment został przeprowadzony dla:
* większej liczby klientów - 8,
* większej ilości obiektów w zestawach treningowych na każdym z klientów – 15000,
* większej liczby rund / iteracji uczenia federacyjnego – 15,
* niezbalansowana dystrybucja obiektów na klientach.

|![chart_3](/images/chart_3.png) | 
|:--:| 
| *Rysunek 9. Dystrybucja klas obiektów w zbiorze uczącym na poszczególnych klientach.* |

|![cmd_3_0](/images/cmd_3.png) | 
|:--:| 
| *Rysunek 10. Uruchomiona instancja serwera biorąca udział w uczeniu federacyjnym (1/2).* |

|![cmd_3_1](/images/cmd_3_1.png) | 
|:--:| 
| *Rysunek 11. Uruchomiona instancja serwera biorąca udział w uczeniu federacyjnym (2/2).* |

|![cmd_3_2](/images/cmd_3_2.png) | 
|:--:| 
| *Rysunek 12. Uruchomione instancje klientów biorących udział w uczeniu federacyjnym (1/3).* |

|![cmd_3_3](/images/cmd_3_3.png) | 
|:--:| 
| *Rysunek 13. Uruchomione instancje klientów biorących udział w uczeniu federacyjnym (2/3).* |

|![cmd_3_4](/images/cmd_3_4.png) | 
|:--:| 
| *Rysunek 14. Uruchomione instancje klientów biorących udział w uczeniu federacyjnym (3/3).* |

Dokładność modelu po stronie poszczególnych klientów w pierwszej rundzie dla zestawu treningowego wahała się od 71.42% do 90.64%. W ostatniej rundzie (15) dokładność dla zestawu walidacyjnego wahała się od 91.17% do 98.88%.

W przypadku dokładności dla zestawu testującego w pierwszej rundzie przyjmowała wartość od 68.25% do 78.35%. W ostatniej rundzie dokładność przyjmowała od 85.04% do 89.03%. 

Zagregowana wersja modelu osiągała dokładność w pierwszej rundzie równą 80.09%. Po wszystkich rundach polepszania (aktualizowania parametrów) modelu globalnego, osiągną on ostatecznie 90.19% dokładności.

## Czwarty eksperyment został przeprowadzony dla:
* większej liczby klientów - 8,
* większej ilości obiektów w zestawach treningowych na każdym z klientów – 15000,
* większej liczby rund / iteracji uczenia federacyjnego – 15,
* zbalansowana dystrybucja obiektów na klientach.

|![chart_4](/images/chart_4.png) | 
|:--:| 
| *Rysunek 15. Dystrybucja klas obiektów w zbiorze uczącym na poszczególnych klientach.* |

|![cmd_4_0](/images/cmd_4.png) | 
|:--:| 
| *Rysunek 16. Uruchomiona instancja serwera biorąca udział w uczeniu federacyjnym (1/2).* |

|![cmd_4_1](/images/cmd_4_1.png) | 
|:--:| 
| *Rysunek 17. Uruchomiona instancja serwera biorąca udział w uczeniu federacyjnym (2/2).* |

|![cmd_4_2](/images/cmd_4_2.png) | 
|:--:| 
| *Rysunek 18. Uruchomione instancje klientów biorących udział w uczeniu federacyjnym (1/3).* |

|![cmd_4_3](/images/cmd_4_3.png) | 
|:--:| 
| *Rysunek 19. Uruchomione instancje klientów biorących udział w uczeniu federacyjnym (2/3).* |

|![cmd_4_4](/images/cmd_4_4.png) | 
|:--:| 
| *Rysunek 20. Uruchomione instancje klientów biorących udział w uczeniu federacyjnym (3/3).* |

Dokładność modelu po stronie poszczególnych klientów w pierwszej rundzie dla zestawu treningowego wahała się od 75.70% do 76.62%. W ostatniej rundzie (15) dokładność dla zestawu walidacyjnego wahała się od 93.03% do 93.50%. 

W przypadku dokładności dla zestawu testującego w pierwszej rundzie przyjmowała wartość od 81.80% do 83.00%. W ostatniej rundzie dokładność przyjmowała od 89.13% do 90.57%. 

Zagregowana wersja modelu osiągała dokładność w pierwszej rundzie równą 82.60%. Po wszystkich rundach polepszania (aktualizowania parametrów) modelu globalnego, osiągną on ostatecznie 91.39% dokładności.

## Podsumowanie
Wszystkie przeprowadzone eksperymenty udało się przeprowadzić bezproblemowo. Udało się zaobserwować jak działa uczenie federacyjne dla różnych wielkości takich czynników jak, ilość danych treningowych, ilość klientów, ilość rund oraz czy dane na poszczególnych instancjach klientów były zróżnicowane pod względem dystrybucji obiektów czy nie. Można było zauważyć jak w kolejnych iteracjach / rundach uczenia federacyjnego system próbuje aktualizować wagi modeli lokalnych tak, aby sukcesywnie zwiększyć ich jakość predykcji, i tym samym aby zagregowany model był jak najdokładniejszy.

Dla pierwszych dwóch eksperymentów korzystających z mniejszej ilości danych treningowych, ilości klientów i ilości rund, można było zauważyć przewidywalną słabszą jakość finalnego globalnego modelu w porównaniu z eksperymentem trzecim i czwartym dla których zwiększono wszystkie z wymienionych czynników.  

W przypadku, gdy dystrybucja obiektów na klientach była niezbalansowana, tj. dany klient posiadał obiekty w znacznej większości z wyznaczonych klas w porównaniu do reszty, jakość dla danych treningowych była bardzo wysoka, lecz jakość dla danych ewaluacyjnych była znacząco niższa (efekt uczenia modelu na niezbalansowanym zbiorze danych). W takich przypadkach federacyjne uczenie ma największą skuteczność. Łączona jest wtedy informacja odnośnie klasyfikacji obiektów jednej klasy na podstawie modelu z jednego klienta z informacjami odnośnie klasyfikacji obiektów innych klas na podstawie modelu z innych klientów. Otrzymujemy wtedy model globalny, który jest wstanie dobrze klasyfikować obiekty z wszystkich klas. 

