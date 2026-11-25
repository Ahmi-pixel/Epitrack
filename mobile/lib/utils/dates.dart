import 'package:intl/intl.dart';

/// Formats a scan's capture date, e.g. "12 Nov 2026".
final scanDateFormat = DateFormat('d MMM y');

/// Whole calendar days from [from] to [to]. The dates are compared as UTC so
/// a daylight-saving change can't shift the count.
int calendarDaysBetween(DateTime from, DateTime to) {
  final start = DateTime.utc(from.year, from.month, from.day);
  final end = DateTime.utc(to.year, to.month, to.day);
  return end.difference(start).inDays;
}
