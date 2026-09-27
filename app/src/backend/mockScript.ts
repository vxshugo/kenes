/**
 * A scripted weekly sync of a 10-person project team in Almaty, as the local recognizer
 * would emit it: lowercase, no punctuation, mostly Russian with Kazakh phrases, a few
 * recognition slips. Who sits where decides the audio source per meeting format (see
 * `MockBackend`): online — everyone but the user is in the call; room — everyone is on the
 * mic; hybrid — "room" people on the mic, "remote" people in the call.
 */
export type Person = {
  id: string;
  name: string;
  site: "room" | "remote";
  /** The user (addressed as «Хуго»). */
  me?: boolean;
};

export type ScriptLine = {
  who: string;
  text: string;
  /** Silence before this line starts, ms. */
  pause: number;
  /** Too short to tell who spoke: the online label is null, the end-of-meeting re-clustering fills it in. */
  unsure?: boolean;
  /** Online clustering mistook the voice for a new speaker; re-clustering merges it back. */
  stray?: boolean;
};

export const MOCK_PEOPLE: Person[] = [
  { id: "me", name: "Хуго", site: "room", me: true },
  { id: "aigerim", name: "Айгерим", site: "room" },
  { id: "erlan", name: "Ерлан", site: "remote" },
  { id: "aidos", name: "Айдос", site: "remote" },
  { id: "dina", name: "Дина", site: "room" },
  { id: "nurlan", name: "Нурлан", site: "remote" },
  { id: "madina", name: "Мадина", site: "room" },
  { id: "timur", name: "Тимур", site: "remote" },
  { id: "asel", name: "Асель", site: "room" },
  { id: "daniyar", name: "Данияр", site: "remote" },
];

export const MOCK_MEETING_TITLE = "Синк по проекту Жолдас";

export const MOCK_MEETING_CONTEXT = `Еженедельный синк по проекту «Жолдас» (приложение доставки, компания «Дала Софт», Алматы).
Участники: Айгерим — проджект-менеджер (ведёт); Ерлан — продакт-оунер со стороны бизнеса; Айдос — мобильная разработка; Дина — QA-лид; Нурлан — DevOps; Мадина — дизайн; Тимур — аналитика; Асель — финансы; Данияр — информационная безопасность (впервые на синке); я — техлид бэкенда.
Повестка:
1. Релиз 2.3 (платёжный шлюз, возвраты) — дата выкатки.
2. SLA по уведомлениям для клиента.
3. Инфраструктура: второй брокер, серверы.
4. Канареечный релиз мобильного приложения.
5. Тестирование и найм.
6. Пентест.
Мои заметки: регресс на стейдже занимает ~1 день; банк-эквайер иногда отвечает до 30 с; в прошлом проекте пентест стоил около 2 млн тенге.`;

export const MOCK_SCRIPT: ScriptLine[] = [
  { who: "aigerim", pause: 800, text: "сәлеметсіздер ме коллеги давайте начнем синк по жолдасу у нас сорок минут" },
  { who: "aigerim", pause: 600, text: "у нас сегодня новый человек данияр из безопасности представься пожалуйста" },
  { who: "daniyar", pause: 1100, text: "всем привет мен данияр ақпараттық қауіпсіздік бөлімінен буду вести пентест" },
  { who: "aigerim", pause: 900, text: "отлично хуго начнем с тебя что по платежному шлюзу успеваем к пятнице" },
  { who: "me", pause: 2600, text: "в целом да осталось закрыть возвраты и прогнать регресс на стейдже" },
  { who: "dina", pause: 900, text: "регресс я могу запустить в среду если стейдж будет стабильный" },
  { who: "madina", pause: 700, text: "согласна" },
  { who: "aigerim", pause: 1200, text: "айдос что по мобилке как канареечный релиз" },
  { who: "aidos", pause: 900, text: "да выкатили на десять процентов пользователей пока без критичных крэшей" },
  { who: "timur", pause: 1000, text: "по метрикам крэш рейт ноль три процента это в пределах нормы" },
  { who: "erlan", pause: 1100, text: "тимур а по конверсии в оплату что видим" },
  { who: "timur", pause: 900, text: "конверсия пока на уровне старой версии нужно ещё дня три данных" },
  { who: "erlan", pause: 1400, text: "хорошо тогда вопрос по sla уведомлений мы клиенту обещали девяносто девять и девять это вообще реально с текущей архитектурой или нет" },
  { who: "me", pause: 3600, text: "честно говоря с одной очередью на кафке будет сложно нужен второй брокер" },
  { who: "nurlan", pause: 1000, text: "второй брокер поднимем у нас есть резерв в алматинском дц" },
  { who: "aigerim", pause: 900, text: "спасибо нурлан" },
  { who: "asel", pause: 1300, text: "хугоға сұрақ бұл қосымша сервер бюджетке қанша қосады" },
  { who: "me", pause: 2400, text: "примерно плюс два сервера в месяц точную цифру скажу до четверга" },
  { who: "asel", pause: 900, text: "жақсы түсінікті жду цифру до четверга" },
  { who: "aidos", pause: 700, text: "ок", unsure: true },
  { who: "dina", pause: 1500, text: "хуго вчера скидывал сценарии возвратов я их посмотрела там всё понятно" },
  { who: "aigerim", pause: 1000, text: "дина сколько тестировщиков нам нужно минимум на релиз" },
  { who: "dina", pause: 900, text: "минимум ещё один иначе регресс растянется на два дня" },
  { who: "erlan", pause: 1000, text: "бізге екі тестировщик керек пе әлде біреуі жеткілікті ме", stray: true },
  { who: "dina", pause: 900, text: "біреуі жеткілікті если автотесты допишем" },
  { who: "madina", pause: 1300, text: "по дизайну экран возвратов готов ссылку скину в чат" },
  { who: "aigerim", pause: 800, text: "мадина рахмет" },
  { who: "aigerim", pause: 1400, text: "коллеги кто может взять ревью макетов возвратов до среды" },
  { who: "aidos", pause: 1000, text: "я возьму у меня как раз окно" },
  { who: "daniyar", pause: 1600, text: "по пентесту предлагаю внешнего подрядчика своими силами не успеем" },
  { who: "asel", pause: 900, text: "бюджет на пентест қанша керек" },
  { who: "aigerim", pause: 1100, text: "хуго сіз не дейсіз сыртқы мердігер керек пе" },
  { who: "me", pause: 2600, text: "да я бы взял внешнего подрядчика по прошлому проекту было около двух миллионов тенге" },
  { who: "daniyar", pause: 1000, text: "да это похоже на рынок я соберу три предложения" },
  { who: "erlan", pause: 1200, text: "иә келісемін только давайте без переносов релиза клиент ждет" },
  { who: "aigerim", pause: 1300, text: "итак фиксируем релиз в пятницу хуго и нурлан считают серверы до четверга дина ищет тестировщика" },
  { who: "aigerim", pause: 900, text: "келесі синк бейсенбіде сау болыңыздар" },
  { who: "me", pause: 900, text: "спасибо всем до встречи" },
];
